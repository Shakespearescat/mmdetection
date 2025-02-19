# dataset settings
dataset_type = 'CocoDataset'
data_root = r'E:\University\data\dataset_new'

# data_root = r'E:\University\data\BuildingVal.v1i.coco-segmentation'
####

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                 'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/instances_field_train2019.json',
        #ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='field_2019/'),
        #data_prefix=dict(img='valid/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='eval/instances_field_eval2019.json',
        #ann_file='valid/_annotations.coco.json',
        #'annotations/instances_val2017.json',
        data_prefix=dict(img='field_2019/'),  ##### dict(img='val2017/'),
        #data_prefix=dict(img='valid/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader
###
"""
test_dataloader = dict(
    batch_size=1,
    num_workers=0,  # 单图推理建议设为 0 避免冲突
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        # 直接指定单张图片路径（示例路径，需替换为你的实际路径）
        data_root=data_root,
        ann_file='',  # 留空或不使用标注文件
        data_prefix=dict(img='zrtx-cx.jpg'),  # 直接指向图片文件
        pipeline=test_pipeline,
        test_mode=True,
        # 如果报错，可尝试使用更简单的 CustomDataset
        # type='CustomDataset',
        # data_prefix=dict(img_path=data_root + 'test_image.jpg'),
    )
)
"""
###

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/test/instances_field_test2019.json',  #'annotations/instances_val2017.json',
    #ann_file=data_root + '/valid/_annotations.coco.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator
###
"""
test_evaluator = dict(
    type='CocoMetric',
    ann_file=None,  # 禁用标注验证
    metric=['bbox', 'segm'],
    format_only=False,
    outfile_prefix='./work_dirs/coco_instance/test_single_image',
    backend_args=backend_args
)
"""
###

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric=['bbox', 'segm'],
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_instance/test')
