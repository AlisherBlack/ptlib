_base_ = ["../_base_/default_runtime.py"]

dataset_type = "s3dis"
data_root = "cache/s3dis-compressed"

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
    use_batch_norm=True,
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
        mix_prob=0.8,
    ),
    train=dict(
        _target_="ptlib.datasets.s3dis.S3DISDataset",
        split=(
            "Area_1",
            "Area_2",
            "Area_3",
            "Area_4",
            "Area_6",
        ),
        data_root=data_root,
        ignore_index=ignore_index,
        transform=[
            dict(_target_="ptlib.datasets.transform.CenterShift", apply_z=True),
            dict(
                _target_="ptlib.datasets.transform.RandomDropout",
                dropout_ratio=0.2,
                dropout_application_ratio=1.0,
            ),
            dict(
                _target_="ptlib.datasets.transform.RandomRotate",
                angle=[-1, 1],
                axis="z",
                center=[0, 0, 0],
                p=0.5,
            ),
            dict(
                _target_="ptlib.datasets.transform.RandomRotate",
                angle=[-1 / 64, 1 / 64],
                axis="x",
                p=0.5,
            ),
            dict(
                _target_="ptlib.datasets.transform.RandomRotate",
                angle=[-1 / 64, 1 / 64],
                axis="y",
                p=0.5,
            ),
            dict(_target_="ptlib.datasets.transform.RandomScale", scale=[0.9, 1.1]),
            dict(_target_="ptlib.datasets.transform.RandomFlip", p=0.5),
            dict(
                _target_="ptlib.datasets.transform.RandomJitter",
                sigma=0.005,
                clip=0.02,
            ),
            # dict(
            #     _target_="ptlib.datasets.transform.ElasticDistortion",
            #     distortion_params=[[0.2, 0.4], [0.8, 1.6]],
            # ), # тяжелая
            dict(
                _target_="ptlib.datasets.transform.ChromaticAutoContrast",
                p=0.2,
                blend_factor=None,
            ),
            dict(
                _target_="ptlib.datasets.transform.ChromaticTranslation",
                p=0.95,
                ratio=0.05,
            ),
            dict(
                _target_="ptlib.datasets.transform.ChromaticJitter",
                p=0.95,
                std=0.05,
            ),
            dict(
                _target_="ptlib.datasets.transform.GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),  # тяжелая
            dict(
                _target_="ptlib.datasets.transform.SphereCrop",
                sample_rate=0.6,
                mode="random",
            ),  # тяжелая
            dict(
                _target_="ptlib.datasets.transform.SphereCrop",
                point_max=204800,
                mode="random",
            ),  # тяжелая
            dict(_target_="ptlib.datasets.transform.CenterShift", apply_z=False),
            dict(_target_="ptlib.datasets.transform.NormalizeColor"),
            dict(_target_="ptlib.datasets.transform.ToTensor"),
            dict(
                _target_="ptlib.datasets.transform.Collect",
                keys=(
                    "coord",
                    "grid_coord",  # добавить, если буду использовать grid_coord
                    "segment",
                ),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        _target_="ptlib.datasets.s3dis.S3DISDataset",
        split="Area_5",
        data_root=data_root,
        ignore_index=ignore_index,
        transform=[
            dict(_target_="ptlib.datasets.transform.CenterShift", apply_z=True),
            dict(
                _target_="ptlib.datasets.transform.SphereCrop",
                sample_rate=0.6,
                mode="random",
            ),  # кроп до GridSample, чтобы inverse был корректен
            dict(
                _target_="ptlib.datasets.transform.SphereCrop",
                point_max=204800,
                mode="random",
            ),  # кроп до GridSample, чтобы inverse был корректен
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
        split="Area_5",
        data_root=data_root,
        ignore_index=ignore_index,
        transform=[
            dict(_target_="ptlib.datasets.transform.CenterShift", apply_z=True),
            dict(_target_="ptlib.datasets.transform.NormalizeColor"),
            dict(
                _target_="ptlib.datasets.transform.Copy",
                keys_dict={"segment": "origin_segment"},
            ),
            dict(
                _target_="ptlib.datasets.transform.GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
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
                [
                    dict(
                        _target_="ptlib.datasets.transform.RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        _target_="ptlib.datasets.transform.RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        _target_="ptlib.datasets.transform.RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        _target_="ptlib.datasets.transform.RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        _target_="ptlib.datasets.transform.RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(
                        _target_="ptlib.datasets.transform.RandomScale",
                        scale=[0.95, 0.95],
                    ),
                ],
                [
                    dict(
                        _target_="ptlib.datasets.transform.RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(
                        _target_="ptlib.datasets.transform.RandomScale",
                        scale=[0.95, 0.95],
                    ),
                ],
                [
                    dict(
                        _target_="ptlib.datasets.transform.RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(
                        _target_="ptlib.datasets.transform.RandomScale",
                        scale=[0.95, 0.95],
                    ),
                ],
                [
                    dict(
                        _target_="ptlib.datasets.transform.RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(
                        _target_="ptlib.datasets.transform.RandomScale",
                        scale=[0.95, 0.95],
                    ),
                ],
                [
                    dict(
                        _target_="ptlib.datasets.transform.RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(
                        _target_="ptlib.datasets.transform.RandomScale",
                        scale=[1.05, 1.05],
                    ),
                ],
                [
                    dict(
                        _target_="ptlib.datasets.transform.RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(
                        _target_="ptlib.datasets.transform.RandomScale",
                        scale=[1.05, 1.05],
                    ),
                ],
                [
                    dict(
                        _target_="ptlib.datasets.transform.RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(
                        _target_="ptlib.datasets.transform.RandomScale",
                        scale=[1.05, 1.05],
                    ),
                ],
                [
                    dict(
                        _target_="ptlib.datasets.transform.RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(
                        _target_="ptlib.datasets.transform.RandomScale",
                        scale=[1.05, 1.05],
                    ),
                ],
                [dict(_target_="ptlib.datasets.transform.RandomFlip", p=1)],
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
