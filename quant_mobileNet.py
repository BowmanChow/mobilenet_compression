# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import copy
import time
from time import gmtime, strftime, perf_counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path
import yaml
from collections import OrderedDict

import nni
from nni.algorithms.compression.pytorch.quantization import (
    DoReFaQuantizer,
    QAT_Quantizer,
    LsqQuantizer,
    BNNQuantizer,
    ObserverQuantizer,
    NaiveQuantizer
)
from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT

from utils import *
torch.cuda.empty_cache()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model_type = 'mobilenet_v2_torchhub'   # 'mobilenet_v1' 'mobilenet_v2' 'mobilenet_v2_torchhub'
pretrained = True                    # load imagenet weight (only for 'mobilenet_v2_torchhub')
log_name_additions = ''
input_size = 224
n_classes = 120

# reduce CPU usage
train_dataset, train_dataloader = None, None
train_dataset_for_pruner, train_dataloader_for_pruner = None, None
valid_dataset, valid_dataloader = None, None
test_dataset, test_dataloader = None, None 

# optimization parameters    (for finetuning)
batch_size = 32
n_epochs = 0
learning_rate = 1e-4         # 1e-4 for finetuning, 1e-3 (?) for training from scratch

import tensorrt as trt

def get_model_size(model_or_engine):
    if isinstance(model_or_engine, torch.nn.Module):
        # 处理普通的 PyTorch 模型
        torch.save(model_or_engine.state_dict(), 'temp_model.pth')
        model_size = os.path.getsize('temp_model.pth') / (1024 * 1024)  # 将字节转换为MB
        os.remove('temp_model.pth')
        return model_size
    elif isinstance(model_or_engine, trt.ICudaEngine):
        # 处理 TensorRT Engine
        with trt.IHostMemory() as serialized_engine:
            engine = model_or_engine
            serialized_engine = engine.serialize()
            engine_size = serialized_engine.size / (1024 * 1024)  # 将字节转换为MB
            return engine_size
    else:
        raise ValueError("Unsupported model or engine type")

def parse_args():
    parser = argparse.ArgumentParser(description='Example code for quant MobileNetV2')
    parser.add_argument('--quan_mode', type=str, default='fp32',help='choose the quan mode for model')
    parser.add_argument('--input_dir', type=str, default='./pretrained_mobilenet_v2_torchhub/')
    parser.add_argument('--input_ckpt_name', type=str, default='checkpoint_best.pt')
    parser.add_argument('--output_dir', type=str, default='./pretrained_mobilenet_v2_torchhub/')
    parser.add_argument('--dataset_dir', type=str, default='./data/stanford-dogs')

    parser.add_argument('--calc_initial_yaml', action='store_true', default=False)
    parser.add_argument('--calc_final_yaml', action='store_true', default=False)

    args = parser.parse_args()
    return args

def run_test(model,device):
    model.eval()
    total_images = 0
    loss_func = nn.CrossEntropyLoss()
    acc_list, loss_list = [], []
    with torch.no_grad():
        start_time_raw = perf_counter()
        total_time = 0
        for i, (inputs, labels) in enumerate(tqdm(test_dataloader)):
            start_time_raw = perf_counter()
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds = model(inputs)
            preds = preds.to(device)
            pred_idx = preds.max(1).indices
            end_time_raw = perf_counter()
            total_time += end_time_raw - start_time_raw
            acc = (pred_idx == labels).sum().item() / labels.size(0)
            acc_list.append(acc)
            loss = loss_func(preds, labels).item()
            loss_list.append(loss)
            total_images += inputs.size(0)  # 计算总图片数
        # end_time_raw = perf_counter()

    # total_time = end_time_raw - start_time_raw
    perimg_time = float(total_time) / float(total_images) *1000
    final_loss = np.array(loss_list).mean()
    final_acc = np.array(acc_list).mean()

    return final_loss, final_acc, total_time, perimg_time


def run_test_trt(engine,device):
    time_elapsed = 0
    total_images = 0
    loss_func = nn.CrossEntropyLoss()
    acc_list, loss_list = [], []
    with torch.no_grad():
        start_time_raw = perf_counter()
        for i, (inputs, labels) in enumerate(tqdm(test_dataloader)):
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds, time = engine.inference(inputs)
            preds = preds.to(device)
            # preds = preds.cuda()
            pred_idx = preds.max(1).indices
            acc = (pred_idx == labels).sum().item() / labels.size(0)
            acc_list.append(acc)
            loss = loss_func(preds, labels).item()
            loss_list.append(loss)
            time_elapsed += time
            total_images += inputs.size(0)  # 计算总图片数
        end_time_raw = perf_counter()

    final_loss = np.array(loss_list).mean()
    final_acc = np.array(acc_list).mean()
    total_time_trt = end_time_raw - start_time_raw
    perimg_time_trt = total_time_trt / total_images *1000
    perimg_time_elapsed_trt = float(time_elapsed) / float(total_images) *1000
    
    return final_loss, final_acc, total_time_trt, time_elapsed, perimg_time_trt, perimg_time_elapsed_trt


def run_validation(model, valid_dataloader,device):
    model.eval()
    
    loss_func = nn.CrossEntropyLoss()
    acc_list, loss_list = [], []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(valid_dataloader)):
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds= model(inputs)
            preds = preds.to(device)
            pred_idx = preds.max(1).indices
            acc = (pred_idx == labels).sum().item() / labels.size(0)
            acc_list.append(acc)
            loss = loss_func(preds, labels).item()
            loss_list.append(loss)

    valid_loss = np.array(loss_list).mean()
    valid_acc = np.array(acc_list).mean()
    
    return valid_loss, valid_acc


def run_finetune(model, log, device, optimizer=None, short_term=False, ):    
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) 

    best_valid_acc = 0.0
    best_model = None
    for epoch in range(n_epochs if not short_term else 1):
        print('Start training epoch {}'.format(epoch))
        loss_list = []
    
        # train
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds = model(inputs)
            preds = preds.to(device)
            # preds = preds.cuda()
            loss = criterion(preds, labels)
            loss_list.append(loss.item())
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            if short_term:
                break
            
        # validation
        valid_loss, valid_acc = run_validation(model, valid_dataloader,device)
        train_loss = np.array(loss_list).mean()
        print('Epoch {}: train loss {:.4f}, valid loss {:.4f}, valid acc {:.4f}\n'.format
              (epoch, train_loss, valid_loss, valid_acc))
        log.write('Epoch {}: train loss {:.4f}, valid loss {:.4f}, valid acc {:.4f}\n'.format
                  (epoch, train_loss, valid_loss, valid_acc))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            #best_model = copy.deepcopy(model).to(device)

    log.write("Best validation accuracy: {}".format(best_valid_acc))

    #model = best_model
    #return model


def trainer_helper(model, criterion, optimizer, device):
    print("Running trainer in tuner")
    for epoch in range(1):
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader_for_pruner)):
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds = model(inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()


quantizer_name_to_class = {
    'dorefa': DoReFaQuantizer,
    'qat': QAT_Quantizer,
    'lsq': LsqQuantizer,
    'bnn': BNNQuantizer,
    'observer': ObserverQuantizer,
    'naive': NaiveQuantizer,
}

            
def main(args, quantizer_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    quantizer_name = 'qat'
    log_name = args.output_dir / f'quantization_{quantizer_name}_{log_name_additions}.log'
    log = open(log_name, 'w')
    def print_log(text: str):
        print(text)
        log.write(f"{text}\n")
    
    checkpoint_path = args.input_dir / args.input_ckpt_name
    model = torch.load(checkpoint_path)
    if isinstance(model, OrderedDict):
        model = create_model(model_type=model_type, pretrained=pretrained, n_classes=n_classes,
                            input_size=input_size, checkpoint=checkpoint_path)

    # 在剪枝+量化之前
    original_model_size = get_model_size(model)
    print_log(f"初始模型 {checkpoint_path} 存储占用大小: {original_model_size:.2f} MB")

    model = model.to(device)
    # evaluation before quantization 
    flops, params = count_flops(model, log, device)
    initial_loss, initial_acc, total_time_bef, perimg_time_bef= run_test(model,device)
    print_log(f"Inference elapsed raw time: {total_time_bef} s")
    print_log(f"Average time per image: {perimg_time_bef} ms")
    print_log(f"Before Quantization: \n Loss: {initial_loss} \n Accuracy: {initial_acc} \n")
    
    mflops = flops/1e6  
    if args.calc_initial_yaml:
        with open(args.output_dir / 'logs.yaml', 'w') as f:
            yaml_data = {
                'Accuracy': {'baseline': round(100*float(initial_acc), 2), 'method': None},
                'FLOPs': {'baseline': round(mflops, 2), 'method': None},
                'Parameters': {'baseline': round(params/1e6, 2), 'method': None},
                'Infer_times': {'baseline': round(total_time_bef, 2), 'method': None},
                'Storage': {'baseline': round(original_model_size, 2), 'method': None},
            }
            yaml.dump(yaml_data, f)
        
    # quantization
    config_list = [{
        'quant_types': ['weight', 'output'],
        'quant_bits': {
            'weight': 16,
            'output': 16
        },
        'op_types':['Conv2d', 'Linear']
    }]

    kwargs = {}
    optimizer = None
    if quantizer_name == 'qat':
        # dummy_input = torch.rand(1, 3, 224, 224).cuda()
        dummy_input = torch.rand(1, 3, 224, 224).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        kwargs['dummy_input'] = dummy_input
        kwargs['optimizer'] = optimizer

    quantizer = quantizer_name_to_class[quantizer_name](model, config_list, **kwargs)
    quantizer.compress()
    # run inference for the quantizer to finalize model
    # run_validation(model, valid_dataloader, device=device)
    # run_finetune(model, log, device=device, short_term=True)
    calibration_config = quantizer.export_model(log_name.with_suffix('.pt'), log_name.with_suffix('.calibration.pt'))
    # print(f"{calibration_config = }")
    
    # finetuning and final evaluation
    # run_finetune(model, log, device)
    final_loss, final_acc, total_time_ft, perimg_time_ft= run_test(model,device)
    print_log(f"Inference elapsed raw time: {total_time_ft} s")
    print_log(f"Average time per image: {perimg_time_ft} ms")
    print_log(f"After Quantization: \n Loss: {final_loss} \n Accuracy: {final_acc} \n")
    count_flops(model, log)

    # # Inference with Speedup
    # if args.quan_mode == "int8":
    #     extra_layer_bit = 8
    # elif args.quan_mode == "fp16":
    #     extra_layer_bit = 16
    # elif args.quan_mode == "best":
    #     extra_layer_bit = -1
    # else:
    #     extra_layer_bit = 32
    
    batch_size = 32
    input_shape = (batch_size, 3, 224, 224)
    engine = ModelSpeedupTensorRT(
        model,
        input_shape,
        config=None,
        calib_data_loader=valid_dataloader,
        batchsize=batch_size,
        extra_layer_bit=16,
        onnx_path=os.path.join(args.output_dir, "temp_engine.onnx"),
    )
    # engine = ModelSpeedupTensorRT(model, input_shape, config=calibration_config, batchsize=batch_size,extra_layer_bit=extra_layer_bit)

    engine.compress()

    # 在量化加速完成后
    # 指定路径并获取 Engine 大小
    engine_path = args.output_dir / "temp_engine.trt"
    engine.export_quantized_model(engine_path)
    engine_size = os.path.getsize(engine_path) / (1024 * 1024)  # 获取文件大小并转换为MB
    # os.remove(engine_path)  # 删除临时文件
    print_log(f"量化加速后模型 {engine_path} 存储占用大小: {engine_size:.2f} MB")

    final_loss, final_acc, total_time_trt, time_elapsed, perimg_time_trt, perimg_time_elapsed_trt = run_test_trt(engine,device)
    print_log(f"Inference elapsed raw time: {total_time_trt} s")
    print_log(f"Inference elapsed_time (calculated by inference engine): {time_elapsed} s")
    print_log(f"Average time per image: {perimg_time_trt} ms")
    print_log(f"Final After Quantization: \n Loss: {final_loss} \n Accuracy: {final_acc} \n")
    flops, params = count_flops(model, log, device)
            
    mflops = flops/1e6
    if args.calc_final_yaml:
        yaml_data = yaml.safe_load(open(args.output_dir / 'logs.yaml', 'r'))
        with open(args.output_dir / 'logs.yaml', 'w') as f:
            yaml_data = {
                'Accuracy': {'baseline': yaml_data['Accuracy']['baseline'], 'method': round(100*float(final_acc), 2)},
                'FLOPs': {'baseline': yaml_data['FLOPs']['baseline'], 'method': round(mflops, 2)},
                'Parameters': {'baseline': yaml_data['Parameters']['baseline'], 'method': round(params/1e6, 2)},
                'Infer_times': {'baseline': yaml_data['Infer_times']['baseline'], 'method': round(time_elapsed, 2)},
                'Storage': {'baseline': yaml_data['Storage']['baseline'], 'method': round(engine_size, 2)},
                'Output_file': str(engine_path),
            }
            yaml.dump(yaml_data, f)
    log.close()


if __name__ == '__main__':
    args = parse_args()
    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(exist_ok=True)
    args.dataset_dir = Path(args.dataset_dir)

    dataset_path = args.dataset_dir
    train_dataset = TrainDataset(str(dataset_path / 'Processed/train'))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataset_for_pruner = EvalDataset(str(dataset_path / 'Processed/train'))
    train_dataloader_for_pruner = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataset = EvalDataset(str(dataset_path / 'Processed/valid'))
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = EvalDataset(str(dataset_path / 'Processed/test'))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    main(args)
    
