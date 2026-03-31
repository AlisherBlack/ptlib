import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, channel=3, use_batch_norm=True):
        super(PointNetEncoder, self).__init__()

        def mlp_block(in_c, out_c, use_bn, use_relu=True):
            layers = [nn.Conv1d(in_c, out_c, 1)]
            if use_bn:
                layers.append(nn.BatchNorm1d(out_c))
            if use_relu:
                layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        # MLP1: C -> 64 -> 64
        self.mlp1 = nn.Sequential(
            mlp_block(channel, 64, use_batch_norm, use_relu=True),
            mlp_block(64, 64, use_batch_norm, use_relu=True),
        )
        # MLP2: 64 -> 128 -> 1024
        self.mlp2 = nn.Sequential(
            mlp_block(64, 128, use_batch_norm, use_relu=True),
            mlp_block(128, 1024, use_batch_norm, use_relu=True),
        )

        self.global_feat = global_feat

    def forward(self, x):
        B, D, N = x.size()

        x = self.mlp1(x)  # C -> 64 -> 64
        pointfeat = x  # save X1 for skip connection

        x = self.mlp2(x)  # 64 -> 128 -> 1024

        x = torch.max(x, 2, keepdim=True)[0]  # MaxPool -> (B, 1024, 1)
        x = x.view(-1, 1024)

        if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1)  # 1024 + 64
