"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SupCEResNet(nn.Module):
    """official Resnet for image classification, e.g., ImageNet"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        self.encoder = models.__dict__[name](pretrained=True)
        self.fc = self.encoder.fc
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        return self.fc(self.encoder(x))


class JointConResNet(nn.Module):
    """Resnet for image classification and contrastive head"""
    def __init__(self, name='resnet50', feat_dim=128, num_classes=10):
        super(JointConResNet, self).__init__()
        self.encoder = models.__dict__[name](pretrained=True)
        self.fc = self.encoder.fc
        self.encoder.fc = nn.Identity()

        dim_in = self.fc.in_features
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim)
        )

    def forward(self, x):
        feat = self.encoder(x)
        logit = self.fc(feat)
        feat = F.normalize(self.head(feat), dim=1)
        return logit, feat
