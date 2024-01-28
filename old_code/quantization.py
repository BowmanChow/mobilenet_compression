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

# os.environ["CUDA_VISIBLE_DEVICES"]="2"
# torch.cuda.set_device(2)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model_type = 'mobilenet_v2_torchhub'   # 'mobilenet_v1' 'mobilenet_v2' 'mobilenet_v2_torchhub'
pretrained = True                    # load imagenet weight (only for 'mobilenet_v2_torchhub')
experiment_dir = './pretrained_mobilenet_v2_torchhub/'
log_name_additions = ''
checkpoint = experiment_dir + '/checkpoint_best.pt'
input_size = 224
n_classes = 120

# reduce CPU usage
train_dataset, train_dataloader = None, None
train_dataset_for_pruner, train_dataloader_for_pruner = None, None
valid_dataset, valid_dataloader = None, None
test_dataset, test_dataloader = None, None 

# optimization parameters    (for finetuning)
batch_size = 32
n_epochs = 3
learning_rate = 1e-4         # 1e-4 for finetuning, 1e-3 (?) for training from scratch


def parse_args():
    parser = argparse.ArgumentParser(description='Example code for quant MobileNetV2')
    parser.add_argument('--quan_mode', type=str, default='fp32',help='choose the quan mode for model')

def run_test(model):
    model.eval()
    total_images = 0
    loss_func = nn.CrossEntropyLoss()
    acc_list, loss_list = [], []
    with torch.no_grad():
        start_time_raw = perf_counter()
        for i, (inputs, labels) in enumerate(tqdm(test_dataloader)):
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds = model(inputs)
            preds = preds.to(device)
            pred_idx = preds.max(1).indices
            acc = (pred_idx == labels).sum().item() / labels.size(0)
            acc_list.append(acc)
            loss = loss_func(preds, labels).item()
            loss_list.append(loss)
            total_images += inputs.size(0)  # 计算总图片数
        end_time_raw = perf_counter()

    print("Inference elapsed raw time: {}s".format(end_time_raw - start_time_raw))
    print("Average time per image: {}ms".format((end_time_raw - start_time_raw) / total_images *1000))  # 计算每张图片的平均时间
    final_loss = np.array(loss_list).mean()
    final_acc = np.array(acc_list).mean()

    return final_loss, final_acc


def run_test_trt(engine):
    time_elapsed = 0
    total_images = 0
    loss_func = nn.CrossEntropyLoss()
    acc_list, loss_list = [], []
    with torch.no_grad():
        start_time_raw = perf_counter()
        for i, (inputs, labels) in enumerate(tqdm(test_dataloader)):
            # inputs, labels = inputs.float().to(device), labels.to(device)
            inputs, labels = inputs.float().to(device), labels.to(device)
            print(type(inputs),type(labels))
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

    print("Inference elapsed raw time: {}s".format(end_time_raw - start_time_raw))
    print("Inference elapsed_time (calculated by inference engine): {}s".format(time_elapsed))
    print("Average time per image: {}ms".format(time_elapsed / total_images *1000))  # 计算每张图片的平均时间
    
    return final_loss, final_acc


def run_validation(model, valid_dataloader):
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


def run_finetune(model, log, optimizer=None, short_term=False):    
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
            loss = criterion(preds, labels)
            loss_list.append(loss.item())
            loss.backward(retain_graph=True)
            optimizer.step()
            if short_term:
                break
            
        # validation
        valid_loss, valid_acc = run_validation(model, valid_dataloader)
        train_loss = np.array(loss_list).mean()
        print('Epoch {}: train loss {:.4f}, valid loss {:.4f}, valid acc {:.4f}'.format
              (epoch, train_loss, valid_loss, valid_acc))
        log.write('Epoch {}: train loss {:.4f}, valid loss {:.4f}, valid acc {:.4f}\n'.format
                  (epoch, train_loss, valid_loss, valid_acc))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            #best_model = copy.deepcopy(model).to(device)

    log.write("Best validation accuracy: {}".format(best_valid_acc))

    #model = best_model
    #return model


def trainer_helper(model, criterion, optimizer):
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
    quantizer_name = 'qat'
    
    log_name = experiment_dir + '/quantization_{}_{}{}.log'.format(quantizer_name, strftime("%Y%m%d%H%M", gmtime()), log_name_additions)
    log = open(log_name, 'w')
    
    model = create_model(model_type=model_type, pretrained=pretrained, n_classes=n_classes,
                         input_size=input_size, checkpoint=checkpoint)
    model = model.to(device)

    # evaluation before quantization 
    count_flops(model, log)
    initial_loss, initial_acc = run_test(model)
    print('Before Quantization:\nLoss: {}\nAccuracy: {}'.format(initial_loss, initial_acc))
    log.write('Before Quantization:\nLoss: {}\nAccuracy: {}\n'.format(initial_loss, initial_acc))
    
    # for name, weight in model.named_parameters():
    #     print(name, weight.max().item(), weight.min().item())
        
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
        # dummy_input = torch.rand(1, 3, 224, 224)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        kwargs['dummy_input'] = dummy_input
        kwargs['optimizer'] = optimizer

    quantizer = quantizer_name_to_class[quantizer_name](model, config_list, **kwargs)
    quantizer.compress()
    # run inference for the quantizer to finalize model
    #run_validation(model, valid_dataloader)
    # run_finetune(model, log, short_term=True)
    calibration_config = quantizer.export_model(log_name.replace('.log', '.pt'), log_name.replace('.log', '_calibration.pt'))
    # print(calibration_config)
    # for name, weight in model.named_parameters():
    #     print(name, weight.max().item(), weight.min().item())

    
    # finetuning and final evaluation
    run_finetune(model, log)
    final_loss, final_acc = run_test(model)
    print('After Quantization:\nLoss: {}\nAccuracy: {}'.format(final_loss, final_acc))
    log.write('After Quantization:\nLoss: {}\nAccuracy: {}'.format(final_loss, final_acc))
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
    engine = ModelSpeedupTensorRT(model, input_shape, config=calibration_config, batchsize=batch_size)
    # engine = ModelSpeedupTensorRT(model, input_shape, config=calibration_config, batchsize=batch_size,extra_layer_bit=extra_layer_bit)
    for name, weight in model.named_parameters():
        print(name, weight.max().item(), weight.min().item())

    engine.compress()

    final_loss, final_acc = run_test_trt(engine)
    print('Final After Quantization:\nLoss: {}\nAccuracy: {}'.format(final_loss, final_acc))
    log.write('Final After Quantization:\nLoss: {}\nAccuracy: {}'.format(final_loss, final_acc))
    count_flops(model, log)
            
    log.close()


if __name__ == '__main__':
    # create here and reuse
    train_dataset = TrainDataset('./data/stanford-dogs/Processed/train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataset_for_pruner = EvalDataset('./data/stanford-dogs/Processed/train')
    train_dataloader_for_pruner = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataset = EvalDataset('./data/stanford-dogs/Processed/valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = EvalDataset('./data/stanford-dogs/Processed/test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    torch.set_num_threads(16)
    
    args = parse_args()
    main(args)
    
