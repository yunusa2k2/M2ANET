# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 20:42:41 2024

@author: Yunus
"""
import torch
import torch.nn as nn
from torchinfo import summary

num_classes = 2

class MHSA(nn.Module):
    def __init__(self, n_dims, width, height, head):
        super(MHSA, self).__init__()
        self.head = head

        self.Q = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.K = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.V = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, head, n_dims // head, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, head, n_dims // head, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        identity = x
        n_batch, C, width, height = x.size()
        q = self.Q(x).view(n_batch, self.head, C // self.head, -1)
        k = self.K(x).view(n_batch, self.head, C // self.head, -1)
        v = self.V(x).view(n_batch, self.head, C // self.head, -1)
        
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.head, C // self.head, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)
        
        out += identity
        return out

class MBConv3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expansion_ratio=6, stride=1, padding=1):
        super(MBConv3, self).__init__()
        
        # Expansion Phase
        expanded_channels = in_channels * expansion_ratio
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(expanded_channels)
        self.expand_activation = nn.ReLU6(inplace=True)
        
        # Depthwise Convolution
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, 
                                         stride=stride, padding=padding, groups=expanded_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        self.depthwise_activation = nn.ReLU6(inplace=True)
        
        # Squeeze and Excitation Phase
        self.se_reduce = nn.Conv2d(expanded_channels, expanded_channels // 16, kernel_size=1)
        self.se_activation = nn.ReLU(inplace=True)
        self.se_expand = nn.Conv2d(expanded_channels // 16, expanded_channels, kernel_size=1)
        self.se_sigmoid = nn.Sigmoid()
        
        # Pointwise Convolution (Linear Projection)
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        # Identity skip connection if input and output shapes are same
        self.use_residual = (in_channels == out_channels) and (stride == 1)
    
    def forward(self, x):
        identity = x
        
        # Expansion Phase
        x = self.expand_activation(self.expand_bn(self.expand_conv(x)))
        
        # Depthwise Convolution
        x = self.depthwise_activation(self.depthwise_bn(self.depthwise_conv(x)))
        
        # Squeeze and Excitation Phase
        squeeze = torch.mean(x, dim=[2, 3], keepdim=True)
        squeeze = self.se_activation(self.se_reduce(squeeze))
        squeeze = self.se_sigmoid(self.se_expand(squeeze))
        x = x * squeeze
        
        # Pointwise Convolution (Linear Projection)
        x = self.project_bn(self.project_conv(x))
        
        # Identity skip connection if possible
        if self.use_residual:
            x += identity
        
        return x

class M2ANET(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(M2ANET, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.block1 = MBConv3(64, 64, kernel_size=3, stride=1)
        self.block2 = MBConv3(64, 128, kernel_size=3, stride=2)
        self.block3 = MBConv3(128, 128, kernel_size=3, stride=1)
        self.block4 = MBConv3(128, 256, kernel_size=3, stride=2)
        
        self.mhsa1 = nn.Sequential(
            MHSA(256, 28, 28, 2),
            nn.Conv2d(256, 512, kernel_size=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
            )

        self.mhsa2 = nn.Sequential(
            MHSA(512, 14, 14, 4),
            nn.Conv2d(512, 1024, kernel_size=1, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU())
        
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Initial convolutional layer
        out = self.relu(self.bn1(self.conv1(x)))
        
        #print(out.shape)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        
        out = self.mhsa1(out)
        out = self.mhsa2(out)
        
        out = self.avgpool(out)
        out = self.dropout(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        
        return out

def model():
    return M2ANET()

model = model()
print(summary(model))