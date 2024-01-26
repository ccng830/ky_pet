import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from models import iresnet
from models import magface

# our modules
from config import config as cfg


# @title MagLinear - header
class MagLinear(torch.nn.Module):
    """
    Parallel fc for Mag loss
    """

    def __init__(self, in_features, out_features, scale=64.0, easy_margin=True):
        super(MagLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.scale = scale
        self.easy_margin = easy_margin

    def forward(self, x, m, l_a, u_a):
        """
        Here m is a function which generate adaptive margin
        """
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(l_a, u_a)
        ada_margin = m(x_norm)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)

        # norm the weight
        weight_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(F.normalize(x), weight_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            mm = torch.sin(math.pi - ada_margin) * ada_margin
            threshold = torch.cos(math.pi - ada_margin)
            cos_theta_m = torch.where(
                cos_theta > threshold, cos_theta_m, cos_theta - mm
            )
        # multiply the scale in advance
        cos_theta_m = self.scale * cos_theta_m
        cos_theta = self.scale * cos_theta

        return [cos_theta, cos_theta_m], x_norm


# @title SoftmaxBuilder - combin backbone & header
class SoftmaxBuilder(nn.Module):
    def __init__(self, backbone, header):
        super(SoftmaxBuilder, self).__init__()
        self.features = backbone
        self.fc = header

        self.l_margin = 0.45
        self.u_margin = 0.8
        self.l_a = 10
        self.u_a = 110

    def _margin(self, x):
        """generate adaptive margin"""
        margin = (self.u_margin - self.l_margin) / (self.u_a - self.l_a) * (
            x - self.l_a
        ) + self.l_margin
        return margin

    def forward(self, x, target):
        x = self.features(x)
        logits, x_norm = self.fc(x, self._margin, self.l_a, self.u_a)
        return logits, x_norm


def train(
    continue_train=True,
    batch_size=16,
    exp_name=None,
    backbone_name=None,
    header_name=None,
    num_of_classes=13135,
    device="cpu",
):
    if exp_name:
        exp_name = exp_name
    else:
        exp_name = 'testing'
    if not continue_train:
        if backbone_name:
            backbone_path = f"../source/{backbone_name}"
        else:
            backbone_path = None
        if header_name:
            header_path = f"../source/{header_name}"
        else:
            header_path = None
    else:
        if backbone_name:
            backbone_path = f"../myexp/{exp_name}/{backbone_name}"
        else:
            backbone_path = None
        if header_name:
            header_path = f"../myexp/{exp_name}/{header_name}"
        else:
            header_path = None
 
    print("backbone_path:", backbone_path)
    print("header_path:", header_path)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = iresnet.iresnet100(num_classes=512)
    if backbone_path:
        backbone.load_state_dict(
            torch.load(backbone_path, map_location=torch.device(device))
        )
    header = MagLinear(512, num_of_classes, scale=64)
    if header_path:
        header.load_state_dict(
            torch.load(header_path, map_location=torch.device(device))
        )
    model = SoftmaxBuilder(backbone=backbone, header=header)
    print("Load model success!")
    return model
