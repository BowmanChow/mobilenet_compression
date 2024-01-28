# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
from utils import create_model, EvalDataset, count_flops


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = 'mobilenet_v2_torchhub'    # 'mobilenet_v1' 'mobilenet_v2' 'mobilenet_v2_torchhub'
pretrained = False                      # load imagenet weight (only for 'mobilenet_v2_torchhub')
checkpoint_dir = './pretrained_{}/'.format(model_type)
checkpoint = checkpoint_dir + '/checkpoint_best.pt'    # model checkpoint produced by pretrain.py
input_size = 224
n_classes = 120
batch_size = 32


def run_test(model, is_trt=False):

    test_dataset = EvalDataset('./data/stanford-dogs/Processed/test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if not is_trt:
        model = model.to(device)
        # print(model)
        # count_flops(model, device=device)
        model.eval()
    loss_func = nn.CrossEntropyLoss()
    acc_list, loss_list = [], []
    time_elasped = 0
    stat = time.time()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_dataloader)):
            if not is_trt:
                inputs, labels = inputs.float().to(device), labels.to(device)
            if is_trt:
                preds, infer_time= model.inference(inputs)
                time_elasped += infer_time
            else:
                preds= model(inputs) 
                time_elasped += (time.time() - stat) 
            pred_idx = preds.max(1).indices
            acc = (pred_idx == labels).sum().item() / labels.size(0)
            acc_list.append(acc)
            loss = loss_func(preds, labels).item()
            loss_list.append(loss)
            

    final_loss = np.array(loss_list).mean()
    final_acc = np.array(acc_list).mean()
    print('Test loss: {}\nTest accuracy: {}\nTotal infer time: {}s'.format(final_loss, final_acc, time_elasped))


if __name__ == '__main__':
    model = create_model(model_type=model_type, pretrained=pretrained, n_classes=n_classes,
                         input_size=input_size, checkpoint=checkpoint)
    run_test(model, False)
    
