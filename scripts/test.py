import os
import time
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import hydra
from mmengine import Config
from loguru import logger

import torch
import torch.nn.functional as F

from ptlib.datasets.datamodule import collate_fn


def intersection_and_union(pred, target, num_classes, ignore_index=-1):
    """Compute per-class intersection, union, target counts (numpy arrays)."""
    pred = pred.copy().reshape(-1)
    target = target.reshape(-1)
    pred[target == ignore_index] = ignore_index
    intersection = pred[pred == target]
    area_intersection = np.bincount(
        intersection[intersection >= 0].astype(int), minlength=num_classes
    )
    area_pred = np.bincount(pred[pred >= 0].astype(int), minlength=num_classes)
    area_target = np.bincount(target[target >= 0].astype(int), minlength=num_classes)
    area_union = area_pred + area_target - area_intersection
    return area_intersection, area_union, area_target


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class SemSegTester:
    """
    Fragment-based semantic segmentation tester.

    Mirrors Pointcept's SemSegTester logic:
      1. For each scene, iterate over pre-built fragment_list (TTA + voxelization).
      2. Accumulate softmax predictions per original point.
      3. Argmax → per-point prediction.
      4. Compute IoU metrics.
    """

    def __init__(self, cfg, model, save_path):
        self.cfg = cfg
        self.model = model
        self.save_path = save_path

        self.num_classes = cfg.num_classes
        self.ignore_index = cfg.ignore_index
        self.class_names = cfg.get("class_names", None)
        if self.class_names is None:
            self.class_names = [str(i) for i in range(self.num_classes)]

        # Build test dataset
        logger.info("=> Building test dataset ...")
        self.test_dataset = hydra.utils.instantiate(cfg.data.test)
        logger.info(f"   Test dataset size: {len(self.test_dataset)}")

    def test(self):
        logger.info(">>>>>>>>>>>>>>>> Start Precise Evaluation >>>>>>>>>>>>>>>>")

        os.makedirs(self.save_path, exist_ok=True)

        batch_time = AverageMeter()
        intersection_meter_sum = np.zeros(self.num_classes)
        union_meter_sum = np.zeros(self.num_classes)
        target_meter_sum = np.zeros(self.num_classes)

        self.model.eval()
        self.model.cuda()

        for idx in range(len(self.test_dataset)):
            start = time.time()

            data_dict = self.test_dataset[idx]

            # In test mode, dataset returns:
            #   segment, name, fragment_list, [origin_segment, inverse]
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")

            pred_save_path = os.path.join(
                self.save_path, "{}_pred.npy".format(data_name)
            )

            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded cached pred.".format(
                        idx + 1, len(self.test_dataset), data_name
                    )
                )
                pred = np.load(pred_save_path)
                if "origin_segment" in data_dict:
                    segment = data_dict["origin_segment"]
            else:
                pred = torch.zeros((segment.size, self.num_classes)).cuda()

                for i in range(len(fragment_list)):
                    input_dict = collate_fn([fragment_list[i]])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)

                    idx_part = input_dict["index"]

                    with torch.no_grad():
                        seg_logits = self.model(input_dict)  # (N, num_classes)
                        pred_part = F.softmax(seg_logits, dim=-1)

                        bs = 0
                        for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            bs = be

                    logger.info(
                        "Test: {}/{}-{}, Fragment: {}/{}".format(
                            idx + 1,
                            len(self.test_dataset),
                            data_name,
                            i + 1,
                            len(fragment_list),
                        )
                    )

                pred = pred.max(1)[1].data.cpu().numpy()

                if "origin_segment" in data_dict:
                    assert "inverse" in data_dict
                    pred = pred[data_dict["inverse"]]
                    segment = data_dict["origin_segment"]

                np.save(pred_save_path, pred)

            # Compute metrics for this scene
            intersection, union, target = intersection_and_union(
                pred, segment, self.num_classes, self.ignore_index
            )
            intersection_meter_sum += intersection
            union_meter_sum += union
            target_meter_sum += target

            # Per-scene metrics
            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            # Running metrics
            m_iou = np.mean(
                intersection_meter_sum / (union_meter_sum + 1e-10)
            )
            m_acc = np.mean(
                intersection_meter_sum / (target_meter_sum + 1e-10)
            )

            batch_time.update(time.time() - start)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {:.3f} ({:.3f}) "
                "Accuracy {:.4f} ({:.4f}) "
                "mIoU {:.4f} ({:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_dataset),
                    segment.size,
                    batch_time.val,
                    batch_time.avg,
                    acc,
                    m_acc,
                    iou,
                    m_iou,
                )
            )

        # Final metrics
        iou_class = intersection_meter_sum / (union_meter_sum + 1e-10)
        accuracy_class = intersection_meter_sum / (target_meter_sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter_sum) / (sum(target_meter_sum) + 1e-10)

        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                mIoU, mAcc, allAcc
            )
        )
        for i in range(self.num_classes):
            logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.class_names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Precise Evaluation <<<<<<<<<<<<<<<<<")

        return dict(mIoU=mIoU, mAcc=mAcc, allAcc=allAcc)


def get_parser():
    """
    TODO:
    1. create scripts/common_cli.py
    2. out commmon args for train/test into scripts/common_cli.py
    3. add change_cfg_by_args like in train.py
    """
    parser = argparse.ArgumentParser(description="Precise Evaluation (SemSeg)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        required=True,
        help="Path to model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Directory to save predictions and results (default: <ckpt_dir>/test_results)",
    )
    return parser.parse_args()


def main():
    assert torch.cuda.is_available(), "CUDA is required for testing"

    args = get_parser()

    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.ckpt_path}")

    cfg = Config.fromfile(args.config)

    # Resolve save path
    if args.save_path is None:
        args.save_path = args.ckpt_path.parent.parent / "test_results"
    args.save_path = Path(args.save_path)
    args.save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Save path: {args.save_path}")

    # Build model
    logger.info("=> Building model ...")
    model = hydra.utils.instantiate(cfg.model)

    # Load checkpoint
    logger.info(f"=> Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, OrderedDict):
        state_dict = ckpt
    else:
        raise RuntimeError(f"Unknown checkpoint format: {type(ckpt)}")

    load_info = model.load_state_dict(state_dict, strict=True)
    logger.info(f"   Missing keys: {load_info.missing_keys}")
    logger.info(f"   Unexpected keys: {load_info.unexpected_keys}")

    if "epoch" in ckpt:
        logger.info(f"   Checkpoint epoch: {ckpt['epoch']}")

    # Run test
    tester = SemSegTester(cfg, model, save_path=str(args.save_path))
    results = tester.test()

    import json

    summary_path = args.save_path / "results.json"
    with open(summary_path, "w") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    logger.info(f"Results saved to {summary_path}")


if __name__ == "__main__":
    main()
