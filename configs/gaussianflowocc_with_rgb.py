_base_ = ['./gaussianflowocc.py']

# NOTE: enable rgb photometric loss
model = dict(
    render_rgb=True
)

# pipeline is a list — MMCV can't partially override it, so redefine in full
train_pipeline = [
    dict(type='PrepareImageInputs', is_train=True, data_config={{_base_.data_config}}, sequential=True),
    dict(type='LoadAnnotationsBEVDepth', bda_aug_conf={{_base_.bda_aug_conf}}, classes={{_base_.class_names}}, is_train=True),
    dict(type='GaussianFlowOcc_GeneratePseudoLabelsHorizon',
         downscale_factor={{_base_.raster_downscale_factor}}, crop_top={{_base_.raster_crop_top}},
         num_frames={{_base_.num_frames}}, grounded_sam_root={{_base_.mask_gt_root}},
         depth_root={{_base_.depth_gt_root}}, temporal_frame_ids={{_base_.temporal_frame_ids}},
         # NOTE: the actual change: load RGB ground truth
         load_rgb=True),
    dict(type='DefaultFormatBundle3D', class_names={{_base_.class_names}}),
    dict(type='Collect3D', keys=['img_inputs', 'gs_gts', 'gs_intrins', 'gs_extrins']),
]

data = dict(
    # NOTE: reduce bs from 4 to 2 to avoid OOM. 
    # TODO: maybe adjust to 3? 
    samples_per_gpu=3,
    train=dict(pipeline=train_pipeline)
    )