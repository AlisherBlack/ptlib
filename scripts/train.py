import copy
import argparse
from pathlib import Path
from collections import OrderedDict
from datetime import timedelta

import hydra
from mmengine import Config
from omegaconf import OmegaConf
from loguru import logger as log

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from ptlib.datasets.datamodule import DataModuleFromConfig


def get_parser():

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--auto-resume", type=str2bool, default=False)
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True)
    parser.add_argument("--resume_from", type=Path, default=None)
    parser.add_argument("--ckpt-path", type=Path, default=None)
    parser.add_argument("--precision", type=int, choices=[16, 32, 64], default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)

    # DDP arguments
    parser.add_argument(
        "--num-nodes", type=int, default=1, help="Number of nodes for DDP"
    )
    parser.add_argument(
        "--devices", type=int, default=1, help="Number of GPUs per node"
    )

    parser.add_argument("--limit-batches", type=float, default=None)
    parser.add_argument("--ddp-timeout", type=int, default=600)
    parser.add_argument("--tb-logs-dir", type=str, default="tb_logs")

    return parser.parse_args()


def change_cfg_by_args(cfg, args):
    if args.lr:
        cfg.model.optim_cfg.lr = args.lr
    if args.batch_size:
        cfg.data.dataloader.batch_size = args.batch_size
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.num_workers is not None:
        cfg.data.dataloader.num_workers = args.num_workers
    if args.early_stopping_patience:
        for i, cb in enumerate(cfg.callbacks):
            if "EarlyStopping" in cb._target_:
                cfg.callbacks[i]["patience"] = args.early_stopping_patience


def main():

    assert torch.cuda.is_available()

    args = get_parser()

    log.info(f"{args = }")

    cfg = Config.fromfile(args.config)

    # --- Model ---

    change_cfg_by_args(cfg, args)

    model = hydra.utils.instantiate(cfg.model)

    # Load checkpoint weights (not for resuming training, just weights)
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif isinstance(ckpt, OrderedDict):
            pass
        else:
            raise RuntimeError(f"Unknown checkpoint format: {type(ckpt)}")
        model.load_state_dict(ckpt, strict=True)
        log.info(f"Loaded model weights from: {args.ckpt_path}")

    # --- Data ---
    data = DataModuleFromConfig(cfg.data)

    # --- Experiment dir ---
    args.exp_dir = Path(args.exp_dir)
    args.exp_dir.mkdir(exist_ok=True, parents=True)

    # --- Logger ---
    tb_logger = TensorBoardLogger(save_dir=args.tb_logs_dir)

    # --- Callbacks ---
    callbacks = []
    for callback_cfg in cfg.callbacks:
        callback_cfg = copy.deepcopy(callback_cfg)

        # Resolve relative dirpath (for ModelCheckpoint)
        if (
            "dirpath" in callback_cfg
            and not Path(callback_cfg["dirpath"]).is_absolute()
        ):
            callback_cfg["dirpath"] = str(args.exp_dir / callback_cfg["dirpath"])

        callback = hydra.utils.instantiate(callback_cfg)
        callbacks.append(callback)

    # --- Trainer kwargs ---
    trainer_kwargs = {}
    if args.limit_batches:
        trainer_kwargs["limit_train_batches"] = args.limit_batches
        trainer_kwargs["limit_val_batches"] = args.limit_batches
        trainer_kwargs["limit_test_batches"] = args.limit_batches

    # Configure DDP strategy with timeout
    if args.devices > 1 or args.num_nodes > 1:
        strategy = DDPStrategy(
            timeout=timedelta(seconds=args.ddp_timeout),
            find_unused_parameters=False,
        )
    else:
        strategy = "auto"

    log.info(OmegaConf.to_yaml(cfg.to_dict()))

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        logger=tb_logger,
        **trainer_kwargs,
    )

    cfg.dump(args.exp_dir / "config.py")

    # --- Auto resume ---
    if args.auto_resume:
        resume_path = args.exp_dir / "checkpoints/last.ckpt"
        if resume_path.exists():
            log.info(f"Auto-resume: {resume_path} exists, resuming training")
            args.resume_from = resume_path
        else:
            log.info(f"Auto-resume enabled but {resume_path} does not exist")

    # --- Run ---
    if args.mode == "train":
        trainer.fit(
            model,
            data,
            ckpt_path=args.resume_from,
        )
    elif args.mode == "test":
        trainer.test(
            model,
            data,
            ckpt_path=args.ckpt_path,
        )


if __name__ == "__main__":
    main()
