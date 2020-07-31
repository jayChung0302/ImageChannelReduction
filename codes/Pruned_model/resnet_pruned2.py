import torch
import torch.nn as nn
import torchvision.models as models
import math


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride = stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        if out.size() == residual.size():
            out += residual
            # print('good')

        # else:
        #     # print(residual.size(), out.size())
        
        out = self.relu(out)

        return out


class MyClass(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(MyClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layers
        self.layer1 = self._make_layer(block, 64, layers[0][0], stride=2, d_bool=layers[0][1])
        self.layer2 = self._make_layer(block, 128, layers[1][0], stride=2, d_bool=layers[1][1])
        self.layer3 = self._make_layer(block, 256, layers[2][0], stride=2, d_bool=layers[2][1])
        self.layer4 = self._make_layer(block, 512, layers[3][0], stride=2, d_bool=layers[3][1])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size = (1, 1)) # Res-41
        # self.avgpool = nn.AvgPool2d(4, stride=1) # Res-32
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1, d_bool=True):
        """layer1 -> block, 64, layers[0], stride=2"""
        downsample = None
        if d_bool:
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion)
                )

        layers = []
        if blocks == 0:
            pass
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


def pruned_ResNet(version=41, herit=False, num_classes=10):

    if version == 41:
        child_net = MyClass(Bottleneck, [(0, False), (4, True), (5, True), (3, True)], num_classes=num_classes)
    elif version == 32:
        print('This version is not supported yet. The model is not same as original paper.')
        child_net = MyClass(Bottleneck, [(1, False), (4, True), (4, True), (1, True)], num_classes=num_classes)
    elif version == 26:
        print('This version is not supported yet. The model is not same as original paper.')
        child_net = MyClass(Bottleneck, [(0, False), (2, False), (5, False), (1, True)], num_classes=num_classes)
    else:
        print('that version of resnet is not supported.')
        return 0

    if herit:
        org_net = models.resnet50(pretrained=True)
        for name, child in child_net.named_children():
            for name2, params in child.named_parameters():  # ex) name : layer0, 1, 2, ... name2 : 1.conv1.weight

                if name == 'fc':
                    break
                if name == 'conv1':
                    child_net.conv1.weight = org_net.conv1.weight
                    continue
                if name == 'bn1':
                    child_net.bn1.weight = org_net.bn1.weight
                    child_net.bn1.bias = org_net.bn1.bias
                    continue
                if name == 'layer2':
                    if name2 == '0.conv1.weight':
                        if version == 32:
                            child_net.layer2[0].conv1 = org_net.layer2[0].conv1
                            print('successfully inherited.')
                            continue
                        else:
                            k = org_net.layer2[0].conv1.weight
                            s = 0
                            for i in range(4):
                                s += k[:, i*64:i*64+64, :, :]
                            avg = s / 4
                            child_net.layer2[0].conv1.weight = nn.Parameter(avg, requires_grad=True)
                            print('successfully inherited.')
                            continue

                    if name2 == '0.downsample.0.weight':
                        continue

                idx = name2.split('.')
                if len(idx) == 3:
                    child_size = getattr(getattr(getattr(child_net, name)[int(idx[0])], idx[1]), idx[2]).size()
                    parent_size = getattr(getattr(getattr(org_net, name)[int(idx[0])], idx[1]), idx[2]).size()

                    if child_size == parent_size:
                        if idx[2] == 'weight':
                            getattr(getattr(child_net, name)[int(idx[0])], idx[1]).weight = getattr(
                                getattr(child_net, name)[int(idx[0])], idx[1]).weight
                        if idx[2] == 'bias':
                            getattr(getattr(child_net, name)[int(idx[0])], idx[1]).bias = getattr(
                                getattr(child_net, name)[int(idx[0])], idx[1]).bias
                    else:
                        print('param size is not matching : ', name, name2, params.size())
                        print()

                elif len(idx) == 4:  # idx3 - downsampling
                    child_size = getattr(getattr(getattr(getattr(child_net, name)[int(idx[0])], idx[1]), idx[2]),
                                         idx[3]).size()
                    parent_size = getattr(getattr(getattr(getattr(org_net, name)[int(idx[0])], idx[1]), idx[2]),
                                          idx[3]).size()

                    if child_size == parent_size:
                        if idx[3] == 'weight':
                            getattr(getattr(getattr(child_net, name)[int(idx[0])], idx[1]), idx[2]).weight = getattr(
                                getattr(getattr(org_net, name)[int(idx[0])], idx[1]), idx[2]).weight
                        if idx[3] == 'bias':
                            getattr(getattr(getattr(child_net, name)[int(idx[0])], idx[1]), idx[2]).bias = getattr(
                                getattr(getattr(org_net, name)[int(idx[0])], idx[1]), idx[2]).bias
                    else:
                        print('param size is not matching : ', name, name2, params.size())
    return child_net

def main():
    # net = MyClass(Bottleneck, [(0, False), (4, True), (5, True), (3, True)], num_classes=10)  # ResNet-41
    # net = MyClass(Bottleneck, [(1, False), (4, True), (4, True), (1, True)], num_classes=10)  # ResNet-32 x(not supported)
    # net = MyClass(Bottleneck, [(0, False), (2, False), (5, False), (1, True)], num_classes=10)   # ResNet-26 x(not supported)

    net = pruned_ResNet(41, True, 10)

    # for name, child in net.named_children():
    #     for name2, params in child.named_parameters():
    #         print(name, name2, params.size())
    sample = torch.randn(3, 224, 224).unsqueeze(0)
    out = net(sample)
    print('output size : ', out.size())
    print(net)


if __name__ == '__main__':
    main()