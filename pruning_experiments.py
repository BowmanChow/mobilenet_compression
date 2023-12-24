# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
import copy
from time import gmtime, strftime, perf_counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

import nni
from nni.compression.pytorch import ModelSpeedup
from nni.algorithms.compression.pytorch.pruning import (
    LevelPruner,
    SlimPruner,
    FPGMPruner,
    TaylorFOWeightFilterPruner,
    L1FilterPruner,
    L2FilterPruner,
    AGPPruner,
    ActivationMeanRankFilterPruner,
    ActivationAPoZRankFilterPruner
)
import logging
from nni.compression.pytorch.speedup import compressor
compressor._logger.setLevel(logging.WARNING)
from nni.compression.pytorch.speedup import compress_modules
compress_modules._logger.setLevel(logging.WARNING)


from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = 'mobilenet_v2_torchhub'
input_size = 224
n_classes = 120

pruner_type_to_class = {'level': LevelPruner,
                        'l1': L1FilterPruner,
                        'l2': L2FilterPruner,
                        'slim': SlimPruner,
                        'fpgm': FPGMPruner,
                        'taylorfo': TaylorFOWeightFilterPruner,
                        'agp': AGPPruner,
                        'mean_activation': ActivationMeanRankFilterPruner,
                        'apoz': ActivationAPoZRankFilterPruner}


def run_eval(model, dataloader, device):
    model.eval()
    total_images = 0
    loss_func = nn.CrossEntropyLoss()
    acc_list, loss_list = [], []
    with torch.no_grad():
        start_time_raw = perf_counter()
        for i, (inputs, labels) in enumerate(tqdm(dataloader)):
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds= model(inputs)
            pred_idx = preds.max(1).indices
            acc = (pred_idx == labels).sum().item() / labels.size(0)
            acc_list.append(acc)
            loss = loss_func(preds, labels).item()
            loss_list.append(loss)
            total_images += inputs.size(0)  # 计算总图片数
        end_time_raw = perf_counter()

    total_time = end_time_raw - start_time_raw
    perimg_time = total_time / total_images *1000
    final_loss = np.array(loss_list).mean()
    final_acc = np.array(acc_list).mean()

    return final_loss, final_acc, total_time, perimg_time


def run_finetune(model, train_dataloader, valid_dataloader, device,
                 n_epochs=2, learning_rate=1e-4, weight_decay=0.0, log=None):    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_valid_acc = 0.0
    best_model = None
    for epoch in range(n_epochs):
        print('Start finetuning epoch {}'.format(epoch))
        loss_list = []

        # train
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds = model(inputs)
            loss = criterion(preds, labels)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            
        # validation
        valid_loss, valid_acc, *_ = run_eval(model, valid_dataloader, device)
        train_loss = np.array(loss_list).mean()
        print('Epoch {}: train loss {:.4f}, valid loss {:.4f}, valid acc {:.4f}'.format
              (epoch, train_loss, valid_loss, valid_acc))
        if log is not None:
            log.write('Epoch {}: train loss {:.4f}, valid loss {:.4f}, valid acc {:.4f}'.format
                      (epoch, train_loss, valid_loss, valid_acc))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model).to(device)

    print("Best validation accuracy: {}".format(best_valid_acc))
    if log is not None:
        log.write("Best validation accuracy: {}".format(best_valid_acc))
    
    model = best_model
    return model


def run_finetune_distillation(student_model, teacher_model, train_dataloader, valid_dataloader, device,
                              alpha, temperature,
                              n_epochs=2, learning_rate=1e-4, weight_decay=0.0, log=None):
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(student_model.parameters(), lr=learning_rate, momentum=0.9)            

    best_valid_acc = 0.0
    best_model = None
    for epoch in range(n_epochs):
        print('Start finetuning with distillation epoch {}'.format(epoch))
        loss_list = []

        # train
        student_model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.to(device)
            with torch.no_grad():
                teacher_preds = teacher_model(inputs)
            
            preds = student_model(inputs)
            soft_loss = nn.KLDivLoss()(F.log_softmax(preds/temperature, dim=1),
                                       F.softmax(teacher_preds/temperature, dim=1))
            hard_loss = F.cross_entropy(preds, labels)
            loss = soft_loss * (alpha * temperature * temperature) + hard_loss * (1. - alpha)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

        # validation
        valid_loss, valid_acc, *_ = run_eval(student_model, valid_dataloader, device)
        train_loss = np.array(loss_list).mean()
        print('Epoch {}: train loss {:.4f}, valid loss {:.4f}, valid acc {:.4f}'.format
              (epoch, train_loss, valid_loss, valid_acc))
        if log is not None:
            log.write('Epoch {}: train loss {:.4f}, valid loss {:.4f}, valid acc {:.4f}'.format
                      (epoch, train_loss, valid_loss, valid_acc))
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(student_model).to(device)

    print("Best validation accuracy: {}".format(best_valid_acc))
    if log is not None:
        log.write("Best validation accuracy: {}".format(best_valid_acc))

    student_model = best_model
    return student_model


def trainer_helper(model, criterion, optimizer, dataloader, device):
    print("Running trainer in tuner")
    for epoch in range(1):
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds = model(inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()


def trainer_helper_with_distillation(model, teacher_model, alpha, temperature, optimizer, dataloader, device):
    print("Running trainer in tuner")
    for epoch in range(1):
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.to(device)
            
            with torch.no_grad():
                teacher_preds = teacher_model(inputs)
            preds = model(inputs)
            soft_loss = nn.KLDivLoss()(F.log_softmax(preds/temperature, dim=1),
                                       F.softmax(teacher_preds/temperature, dim=1))
            hard_loss = F.cross_entropy(preds, labels)
            loss = soft_loss * (alpha * temperature * temperature) + hard_loss * (1. - alpha)
            loss.backward()
            optimizer.step()


def parse_args():
    parser = argparse.ArgumentParser(description='Example code for pruning MobileNetV2')

    parser.add_argument('--experiment_dir', type=str, default='./pretrained_mobilenet_v2_torchhub',
                        help='directory containing the pretrained model')
    parser.add_argument('--input_dir', type=str, default='./pretrained_mobilenet_v2_torchhub',
                        help='directory containing the input model')
    parser.add_argument('--output_dir', type=str, default='./pretrained_mobilenet_v2_torchhub',
                        help='directory to output')
    parser.add_argument('--dataset_dir', type=str, default='./data/stanford-dogs',
                        help='directory of dataset')
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint_best.pt',
                         help='checkpoint of the pretrained model')
    
    # pruner
    parser.add_argument('--pruning_mode', type=str, default='conv1andconv2',
                        choices=['conv0', 'conv1', 'conv2', 'conv1andconv2', 'all'])
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='target sparsity')
    parser.add_argument('--pruner_name', type=str, default='l1',
                        choices=['l1', 'l2', 'slim', 'agp',
                                 'fpgm', 'mean_activation', 'apoz', 'taylorfo'],
                        help='pruner to use')
    # for agp only
    parser.add_argument('--agp_pruning_alg', default='l1',
                        choices=['l1', 'l2', 'slim', 'fpgm',
                                 'mean_activation', 'apoz', 'taylorfo'],
                        help='pruner to use for agp')
    parser.add_argument('--agp_n_iters', type=int, default=64,
                        help='number of iterations for agp')
    parser.add_argument('--agp_n_epochs_per_iter', type=int, default=1,
                        help='number of epochs per iteration for agp')

    # speed-up
    parser.add_argument('--speed_up', action='store_true', default=False,
                        help='Whether to speed-up the pruned model')

    # finetuning parameters
    parser.add_argument('--n_workers', type=int, default=16,
                        help='number of threads')
    parser.add_argument('--finetune_epochs', type=int, default=180,
                        help='number of epochs to finetune the model')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training and inference')
    parser.add_argument('--kd', action='store_true', default=False,
                        help='Whether to use knowledge distillation')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='Alpha for knowledge distillation loss')
    parser.add_argument('--temp', type=float, default=8,
                        help='Temperature for knowledge distillation loss')

    args = parser.parse_args()
    return args


def run_pruning(args):
    print(args)
    torch.set_num_threads(args.n_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log = open(args.output_dir + '/pruning_{}_{}_sparsity{}_{}.log'.format(
        args.pruner_name, args.pruning_mode, args.sparsity,
        strftime("%Y%m%d%H%M", gmtime())), 'w')
    def print_log(text: str):
        print(text)
        log.write(text)
    
    dataset_path = Path(args.dataset_dir) / "Processed"
    print_log(f"Building Dataset in {dataset_path}")
    train_dataset = TrainDataset(str(dataset_path / 'train'))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_dataset_for_pruner = EvalDataset(str(dataset_path / 'train'))
    train_dataloader_for_pruner = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    valid_dataset = EvalDataset(str(dataset_path / 'valid'))
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset = EvalDataset(str(dataset_path / 'test'))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_model_path = os.path.join(args.input_dir, args.checkpoint_name)
    model = create_model(model_type=model_type, pretrained=False, n_classes=n_classes,
                         input_size=input_size, checkpoint=input_model_path)
    print_log(f"Load model from {input_model_path}")
    model = model.to(device)
    original_model_size = os.path.getsize(input_model_path) / (1024 * 1024)  # 将字节转换为MB
    print_log(f"\n剪枝之前模型 {input_model_path} 的存储占用大小: {original_model_size:.2f} MB\n")

    teacher_model = None
    if args.kd:
        teacher_model = copy.deepcopy(model)

    # evaluation before pruning
    # count_flops(model, log, device)
    initial_loss, initial_acc, total_time_bef, perimg_time_bef = run_eval(model, test_dataloader, device)
    print_log(f"\nInference elapsed raw time: {total_time_bef} s")
    print_log(f"Average time per image: {perimg_time_bef} ms")
    print_log(f"Before Pruning:\nLoss: {initial_loss}\nAccuracy: {initial_acc}\n")

    # set up config list and pruner
    config_list = []
    if 'conv0' in args.pruning_mode or args.pruning_mode == 'all':
        if args.pruner_name == 'slim' or (args.pruner_name == 'agp' and args.agp_pruning_alg == 'slim'):
            config_list.append({
                'op_names': ['features.{}.conv.0.1'.format(x) for x in range(2, 18)],
                'sparsity': args.sparsity
            })
        else:
            config_list.append({
                'op_names': ['features.{}.conv.0.0'.format(x) for x in range(2, 18)],
                'sparsity': args.sparsity
            })
    if 'conv1' in args.pruning_mode or args.pruning_mode == 'all':
        if args.pruner_name == 'slim' or (args.pruner_name == 'agp' and args.agp_pruning_alg == 'slim'):
            config_list.append({
                'op_names': ['features.{}.conv.1.1'.format(x) for x in range(2, 18)],
                'sparsity': args.sparsity
            })
        else:
            config_list.append({
                'op_names': ['features.{}.conv.1.0'.format(x) for x in range(2, 18)],
                'sparsity': args.sparsity
            })
    if 'conv2' in args.pruning_mode or args.pruning_mode == 'all':
        if args.pruner_name == 'slim' or (args.pruner_name == 'agp' and args.agp_pruning_alg == 'slim'):
            config_list.append({
                'op_names': ['features.{}.conv.3'.format(x) for x in range(2, 18)],
                'sparsity': args.sparsity
            })
        else:
            config_list.append({
                'op_names': ['features.{}.conv.2'.format(x) for x in range(2, 18)],
                'sparsity': args.sparsity
            })
    print(config_list)

    kwargs = {}
    if args.pruner_name in ['slim', 'taylorfo', 'mean_activation', 'apoz', 'agp']:
        def trainer(model, optimizer, criterion, epoch):
            if not args.kd:
                return trainer_helper(model, criterion, optimizer, train_dataloader, device)
            else:
                return trainer_helper_with_distillation(model, teacher_model, args.alpha, args.temp, optimizer, train_dataloader, device)
        kwargs = {
            'trainer': trainer,
            'optimizer': torch.optim.Adam(model.parameters()),
            'criterion': nn.CrossEntropyLoss()
        }
        if args.pruner_name  == 'agp':
            kwargs['pruning_algorithm'] = args.agp_pruning_alg
            kwargs['num_iterations'] = args.agp_n_iters
            kwargs['epochs_per_iteration'] = args.agp_n_epochs_per_iter
        if args.pruner_name == 'slim':
            kwargs['sparsifying_training_epochs'] = 10

    # pruning
    model_temp_path = os.path.join(args.output_dir, 'model_temp.pth')
    mask_temp_path = os.path.join(args.output_dir, 'mask_temp.pth')
    pruner = pruner_type_to_class[args.pruner_name](model, config_list, **kwargs)
    pruner.compress()
    pruner.export_model(model_temp_path, mask_temp_path)

    # model speedup
    pruner._unwrap_model()
    if args.speed_up:
        dummy_input = torch.rand(1,3,224,224).to(device)
        ms = ModelSpeedup(model, dummy_input, args.experiment_dir + './mask_temp.pth')
        ms.speedup_model()
        print(model)
        count_flops(model, log)

    intermediate_loss, intermediate_acc, total_time_befft, perimg_time_befft = run_eval(model, test_dataloader, device)
    print_log(f"\nInference elapsed raw time: {total_time_befft} s")
    print_log(f"Average time per image: {perimg_time_befft} ms")
    print_log(f"Before Finetuning:\nLoss: {intermediate_loss}\nAccuracy: {intermediate_acc}\n")

    # finetuning
    if args.kd:
        model = run_finetune_distillation(model, teacher_model, train_dataloader, valid_dataloader, device,
                                          args.alpha, args.temp, n_epochs=args.finetune_epochs,
                                          learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    else:
        model = run_finetune(model, train_dataloader, valid_dataloader, device, n_epochs=args.finetune_epochs,
                             learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        
    # final evaluation
    final_loss, final_acc,total_time_final, perimg_time_final= run_eval(model, test_dataloader, device)
    print_log(f"\nInference elapsed raw time: {total_time_final} s")
    print_log(f"Average time per image: {perimg_time_final} ms")
    print_log(f"After Pruning:\nLoss: {final_loss}\nAccuracy: {final_acc}\n")

    # 在剪枝之后
    pruned_model_path = os.path.join(args.output_dir, 'temp_pruned_model.pth')
    torch.save(model, pruned_model_path) 
    pruned_model_size = os.path.getsize(pruned_model_path) / (1024 * 1024)  # 将字节转换为MB
    # os.remove('temp_pruned_model.pth')
    print_log(f"剪枝之后模型 {pruned_model_path} 的存储占用大小: {pruned_model_size:.2f} MB\n")

    # clean up
    filePaths = [model_temp_path, mask_temp_path]
    for f in filePaths:
        if os.path.exists(f):
            os.remove(f)

    log.close()
    
    
if __name__ == '__main__':
    args = parse_args()
    run_pruning(args)
