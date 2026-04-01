import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import pytorch_lightning as pl
import hydra

from ptlib.models.pointnet.pointnet_utils import PointNetEncoder
from ptlib.models.utils.structure import Point


def input_dict_to_batched(input_dict):
    """
    feat:   (N_total, C)
    offset: (B,)

    convert to batched format using zero-padding

    Returns:
        x:      (B, C, N_max)
        counts: (B,)
    """
    point = Point(input_dict)
    feat = point.feat  # (N_total, C)
    offset = point.offset  # (B,)

    counts = torch.diff(torch.cat([offset.new_zeros(1), offset]))  # (B,)
    batch_size = counts.shape[0]
    C = feat.shape[1]
    N_max = counts.max().int().item()

    x = feat.new_zeros(batch_size, C, N_max)
    start = 0
    for i in range(batch_size):
        n = counts[i].int().item()
        x[i, :, :n] = feat[start : start + n].transpose(0, 1)
        start += n

    return x, counts


def intersection_and_union(output, target, num_classes, ignore_index=-1):
    """
    Compute intersection, union, target per class (as in Pointcept).
    output, target: (N,) long tensors
    """
    output = output.clone().view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(
        intersection.float(), bins=num_classes, min=0, max=num_classes - 1
    )
    area_output = torch.histc(
        output.float(), bins=num_classes, min=0, max=num_classes - 1
    )
    area_target = torch.histc(
        target.float(), bins=num_classes, min=0, max=num_classes - 1
    )
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class PointNet(nn.Module):
    def __init__(self, num_class, channel=9, use_batch_norm=True):
        super(PointNet, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(
            global_feat=False, channel=channel, use_batch_norm=use_batch_norm
        )

        def mlp_block(in_c, out_c, use_bn):
            layers = [nn.Conv1d(in_c, out_c, 1)]
            if use_bn:
                layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        self.mlp1 = mlp_block(1088, 512, use_batch_norm)
        self.mlp2 = mlp_block(512, 256, use_batch_norm)
        self.mlp3 = mlp_block(256, 128, use_batch_norm)
        self.conv4 = nn.Conv1d(128, self.k, 1)

    def forward(self, x):
        batchsize = x.size(0)
        n_pts = x.size(2)

        x = self.feat(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.conv4(x)

        x = x.transpose(2, 1).contiguous()
        x = x.view(batchsize, n_pts, self.k)
        return x


class LitPointNet(pl.LightningModule):
    def __init__(
        self,
        in_channels=9,  # coord, color, normal
        num_classes=13,
        use_batch_norm=True,
        ignore_index=-1,
        optim_cfg=None,
        class_names=None,
        log_cfg=None,
    ):
        super().__init__()
        self.save_hyperparameters()  # saves all __init__ parameters as hyperparameters
        self.backbone = PointNet(
            num_class=num_classes,
            channel=in_channels,
            use_batch_norm=use_batch_norm,
        )
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.optim_cfg = optim_cfg
        self.class_names = class_names
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        self.log_cfg = log_cfg or {}

        # accumulators for val metrics
        self._val_intersection = []
        self._val_union = []
        self._val_target = []

    def forward(self, input_dict):
        x, counts = input_dict_to_batched(input_dict)
        logits_batched = self.backbone(x)

        # Unpad back to flat (N_total, num_classes)
        parts = []
        for i in range(logits_batched.shape[0]):
            n = counts[i].int().item()
            parts.append(logits_batched[i, :n, :])
        seg_logits = torch.cat(parts, dim=0)
        return seg_logits

    def _get_log_params(self, metric_name: str, stage: str, batch_size: int):
        log_on_step = stage == "train"
        defaults = {
            "on_step": log_on_step,
            "on_epoch": True,
            "prog_bar": False,
            "sync_dist": True,
            "batch_size": batch_size,
        }
        per_metric = self.log_cfg.get(metric_name, {})
        params = {**defaults, **per_metric}  # {a, b}, b rewrites a
        return params

    def _log_losses_and_metrics(
        self, loss_dict, metrics_dict=None, stage="train", batch_size=1
    ):
        for key, value in loss_dict.items():
            name = f"{stage}_{key}"
            self.log(name, value, **self._get_log_params(name, stage, batch_size))

        if metrics_dict:
            for key, value in metrics_dict.items():
                name = f"{stage}_{key}"
                self.log(name, value, **self._get_log_params(name, stage, batch_size))

    def training_step(self, batch, batch_idx):
        seg_logits = self.forward(batch)  # (N_total, num_classes)
        loss = self.loss_fn(seg_logits, batch["segment"])

        self._log_losses_and_metrics(
            loss_dict=dict(loss_ce=loss), stage="train", batch_size=1
        )
        return loss

    def validation_step(self, batch, batch_idx):

        seg_logits = self.forward(batch)  # (N_total, num_classes)
        loss = self.loss_fn(seg_logits, batch["segment"])

        self._log_losses_and_metrics(loss_dict=dict(CE=loss), stage="val", batch_size=1)

        # inverse mapping if present (grid-sampled val)
        segment = batch["segment"]
        if "inverse" in batch:
            pred = seg_logits.argmax(dim=1)[batch["inverse"]]
            segment = batch["origin_segment"]
        else:
            pred = seg_logits.argmax(dim=1)

        i, u, t = intersection_and_union(
            pred, segment, self.num_classes, self.ignore_index
        )
        self._val_intersection.append(i)
        self._val_union.append(u)
        self._val_target.append(t)

    def on_validation_epoch_end(self):
        if not self._val_intersection:
            raise RuntimeError

        intersection = torch.stack(self._val_intersection).sum(dim=0).cpu().numpy()
        union = torch.stack(self._val_union).sum(dim=0).cpu().numpy()
        target = torch.stack(self._val_target).sum(dim=0).cpu().numpy()

        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)

        # Only average over classes that are present in the ground truth
        present_mask = target > 0
        assert present_mask.any()  # always there are labels
        m_iou = np.mean(iou_class[present_mask])
        m_acc = np.mean(acc_class[present_mask])
        all_acc = sum(intersection) / (sum(target) + 1e-10)

        metrics = dict(mIoU=m_iou, mAcc=m_acc, allAcc=all_acc)
        for i in range(self.num_classes):
            name = self.class_names[i] if self.class_names else str(i)
            metrics[f"IoU_{name}"] = iou_class[i]
            metrics[f"Acc_{name}"] = acc_class[i]

        self._log_losses_and_metrics(
            loss_dict={}, metrics_dict=metrics, stage="val", batch_size=1
        )

        self._val_intersection.clear()
        self._val_union.clear()
        self._val_target.clear()

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optim_cfg, params=self.parameters())
        return optimizer


