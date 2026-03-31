_base_ = ["../_base_/default_runtime.py"]

dataset_type = "s3dis"
data_root = "/home/alisherblack/stuff/sonata/cache/s3dis-compressed-sanity-check"

num_classes = 13
ignore_index = -1

class_names = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
]

model = dict(
    _target_="ptlib.models.pointnet.pointnet_sem_seg.LitPointNet",
    _recursive_=False,  # чтобы optim_cfg инициализровался в нужном месте
    in_channels=9,  # coord(3) + color(3) + normal(3)
    num_classes=num_classes,
    use_batch_norm=False,  # BN is unstable with batch_size=1 sanity check
    ignore_index=ignore_index,
    class_names=class_names,
    optim_cfg=dict(
        _target_="torch.optim.AdamW",
        lr=3e-4,
        weight_decay=0.01,
    ),
    log_cfg=dict(
        train_loss_ce=dict(
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        ),
        val_CE=dict(
            on_epoch=True,
            prog_bar=True,
        ),
        val_mIoU=dict(
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        ),
        val_mAcc=dict(
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        ),
        val_allAcc=dict(
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        ),
    ),
)

data = dict(
    dataloader=dict(
        batch_size=4,
        num_workers=4,
    ),
    train=dict(
        _target_="ptlib.datasets.s3dis.S3DISDataset",
        split=("Area_1",),
        data_root=data_root,
        ignore_index=ignore_index,
        transform=[
            dict(_target_="ptlib.datasets.transform.CenterShift", apply_z=True),
            # NOTE: all random augmentations removed for sanity-check
            # so that train sees the exact same data every epoch.
            dict(
                _target_="ptlib.datasets.transform.GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(_target_="ptlib.datasets.transform.CenterShift", apply_z=False),
            dict(_target_="ptlib.datasets.transform.NormalizeColor"),
            dict(_target_="ptlib.datasets.transform.ToTensor"),
            dict(
                _target_="ptlib.datasets.transform.Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        _target_="ptlib.datasets.s3dis.S3DISDataset",
        split="Area_1",
        data_root=data_root,
        ignore_index=ignore_index,
        transform=[
            dict(_target_="ptlib.datasets.transform.CenterShift", apply_z=True),
            dict(
                _target_="ptlib.datasets.transform.Copy",
                keys_dict={"segment": "origin_segment"},
            ),
            dict(
                _target_="ptlib.datasets.transform.GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(_target_="ptlib.datasets.transform.CenterShift", apply_z=False),
            dict(_target_="ptlib.datasets.transform.NormalizeColor"),
            dict(_target_="ptlib.datasets.transform.ToTensor"),
            dict(
                _target_="ptlib.datasets.transform.Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        _target_="ptlib.datasets.s3dis.S3DISDataset",
        split="Area_1",
        data_root=data_root,
        ignore_index=ignore_index,
        transform=[
            dict(_target_="ptlib.datasets.transform.CenterShift", apply_z=True),
            dict(_target_="ptlib.datasets.transform.NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                _target_="ptlib.datasets.transform.GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(_target_="ptlib.datasets.transform.CenterShift", apply_z=False),
                dict(_target_="ptlib.datasets.transform.ToTensor"),
                dict(
                    _target_="ptlib.datasets.transform.Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "color", "normal"),
                ),
            ],
            aug_transform=[
                # [
                #     dict(
                #         _target_="ptlib.datasets.transform.RandomScale",
                #         scale=[0.9, 0.9],
                #     )
                # ],
                # [
                #     dict(
                #         _target_="ptlib.datasets.transform.RandomScale",
                #         scale=[0.95, 0.95],
                #     )
                # ],
                # [dict(_target_="ptlib.datasets.transform.RandomScale", scale=[1, 1])],
                # [
                #     dict(
                #         _target_="ptlib.datasets.transform.RandomScale",
                #         scale=[1.05, 1.05],
                #     )
                # ],
                # [
                #     dict(
                #         _target_="ptlib.datasets.transform.RandomScale",
                #         scale=[1.1, 1.1],
                #     )
                # ],
                # [
                #     dict(
                #         _target_="ptlib.datasets.transform.RandomScale",
                #         scale=[0.9, 0.9],
                #     ),
                #     dict(_target_="ptlib.datasets.transform.RandomFlip", p=1),
                # ],
                # [
                #     dict(
                #         _target_="ptlib.datasets.transform.RandomScale",
                #         scale=[0.95, 0.95],
                #     ),
                #     dict(_target_="ptlib.datasets.transform.RandomFlip", p=1),
                # ],
                # [
                #     dict(_target_="ptlib.datasets.transform.RandomScale", scale=[1, 1]),
                #     dict(_target_="ptlib.datasets.transform.RandomFlip", p=1),
                # ],
                # [
                #     dict(
                #         _target_="ptlib.datasets.transform.RandomScale",
                #         scale=[1.05, 1.05],
                #     ),
                #     dict(_target_="ptlib.datasets.transform.RandomFlip", p=1),
                # ],
                # [
                #     dict(
                #         _target_="ptlib.datasets.transform.RandomScale",
                #         scale=[1.1, 1.1],
                #     ),
                #     dict(_target_="ptlib.datasets.transform.RandomFlip", p=1),
                # ],
            ],
        ),
    ),
)


monitor = "val_CE"
monitor_mode = "min"

"""
TODO:
1. add SemSegEvaluator
1. add PreciseEvaluator
"""
callbacks = [
    dict(
        _target_="pytorch_lightning.callbacks.ModelCheckpoint",
        dirpath="checkpoints",  # Relative to exp_dir
        monitor=monitor,
        mode=monitor_mode,
        save_top_k=1,  # Save best model
        save_last=True,  # Save last model
        filename=f"best-{{epoch:02d}}-{{{monitor}:.4f}}",
        save_weights_only=False,
        save_on_train_epoch_end=False,
    ),
    dict(
        _target_="ptlib.callbacks.delayed_early_stopping.DelayedEarlyStopping",
        start_after=0,
        count_mode="epoch",
        strict=True,
        monitor=monitor,
        min_delta=0.0001,
        patience=50,
        mode=monitor_mode,
        verbose=True,
        check_on_train_epoch_end=False,
    ),
]
