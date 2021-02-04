_base_ = [
    '../_base_/models/ssd300vital.py', '../_base_/datasets/vkitti.py',
    '../_base_/default_runtime.py'
]
model = dict(
    bbox_head=dict(
        num_classes=6))

# dataset settings
dataset_type = 'VKITTIDataset'
data_root = 'data/vkitti/'
test_dataset_type = 'KITTIDataset'
test_data_root = 'data/kitti/voc/VOC2012/'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=6,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'ImageSets/Main/trainval.txt',
            img_prefix=data_root,
            pipeline=train_pipeline)),
    val=dict(
        type=test_dataset_type,
        ann_file=test_data_root + 'ImageSets/Main/test.txt',
        img_prefix=test_data_root,
        pipeline=test_pipeline),
    test=dict(
        type=test_dataset_type,
        ann_file=test_data_root + 'ImageSets/Main/test.txt',
        img_prefix=test_data_root,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 20])
checkpoint_config = dict(interval=1)
# runtime settings
total_epochs = 24
