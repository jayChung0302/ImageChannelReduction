from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
import visdom

import os
import sys
import copy
import csv
import time
from tqdm import tqdm
import argparse
from Pruned_model.resnet_pruned2 import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Dataset directory', default='/dataset/val')
parser.add_argument('--digit', help='Face emotion database or Street view house number '
                                      '(default: Face emotion)', action='store_true')
parser.add_argument('--rgb', help='3ch rgb or gray (default: gray conversion)', action='store_true')
parser.add_argument('--model_dir', type=str, help='loaded model directory',
                    default='models/FirstLayerFilterpruning/RAF-DB/Y/VGG16_finetuning_conv_1channel.pth')
parser.add_argument('--batch_size', type=int, help='batch size', default=64)

args = parser.parse_args()


def main():
    net = torch.load(args.model_dir)
    if not args.digit:
        if not args.rgb:
            data_transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Grayscale(num_output_channels=1),  # Converts 3channel RGB to 1 channel Gray images
                    transforms.ToTensor(),
                ])

        else:
            data_transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])
    else:
        if not args.rgb:
            data_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=1),  # Converts 3channel RGB to 1 channel Gray images
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4358], std=[0.1229]),
            ])

        else:
            data_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4309, 0.4302, 0.4463], std=[0.1254, 0.1282, 0.1152]),
            ])

    image_datasets = datasets.ImageFolder(args.data_dir, data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=args.batch_size,
                                              shuffle=True, num_workers=4)
    dataset_sizes = len(image_datasets)
    class_names = image_datasets.classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = net.to(device)

    since = time.time()

    print('\n===> Testing Start')
    print('validation data size : ', dataset_sizes)

    # Each epoch has a training and validation phase

    net.eval()

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in tqdm(dataloaders):
        inputs = inputs.to(device)  # load to cuda
        labels = labels.to(device)

        # forward
        with torch.no_grad():
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)
        # statistics
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes
    epoch_acc = running_corrects.double() / dataset_sizes

    # training finished.
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Test Acc: {:4f}'.format(epoch_acc))


if __name__ == '__main__':
    main()