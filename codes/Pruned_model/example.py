import torch
from resnet_pruned2 import *


def main():
    # net = torch.load('C:/Users/lenovo/Dropbox/LAB/DIPL_18/2019-project-4/imageFormatConversion/ResNet41_IMGNT_.pth')
    net = pruned_ResNet(41, True, 10)
    for name, child in net.named_children():
        for name2, params in child.named_parameters():
            print(name, name2, params.size())

if __name__ == '__main__':
    main()