#this is from the coco_instance.py file from the /configs/_base_/datasets folder
# dataset settings
dataset_type = 'CocoDataset' #this is fine the annotationsn are set for this
data_root = '' #enter the correct path for the train data set here. 


backend_args = None


from .coco import build as build_coco #have .coco and .cocoeval
from .micrograph import build as build_micrograph #have .micrograph and .transfomr there
import .misc as utils #also need to import utils.misc
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader


#init the functions that are used later: (note will also have to import the micrographs and transforms.py files that are in this folder)

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args == 'coco':
        return build_coco(image_set, args)
    if args == 'micrograph':
        return build_micrograph(image_set, args)    

    raise ValueError(f'dataset {args} not supported')



"""

I don't think I need this because it is prewrtiten in the datalaoder
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
"""

dataset_train = build_dataset(image_set='train', args="micrograph")
dataset_val = build_dataset(image_set='val', args="micrograph")

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)
batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size=16, drop_last=True)


train_dataloader = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=0)
val_dataloader = DataLoader(dataset_val, batch_size=16, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=0)
test_dataloader = val_dataloader


'''train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
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
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader'''

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

