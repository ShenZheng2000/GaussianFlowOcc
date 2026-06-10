# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import os.path as osp
import mmcv
import gc
import numpy as np

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import Metric_mIoU, IoU

colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])

@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):

    def __init__(self, *args, eval_threshold_range=[.05, .2, .5], num_classes=18,
                 gt_root='data/gts', gt_root2='data/gtsv2', eval_metrics = ['mIoU'],
                 with_others=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.eval_threshold_range = eval_threshold_range
        self.gt_root=gt_root
        self.gt_root_v2=gt_root2
        self.num_classes = num_classes
        self.eval_metrics = eval_metrics
        self.with_others = with_others
        self.v1_to_v2_map = np.array([16, 9, 5, 3, 0, 4, 6, 7, 8, 2, 1, 10, 11, 12, 13, 14, 15])

    def get_data_info(self, index):
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        return input_dict
    
    def evaluate_mIoU(self, occ_results, eval_dict):
        semantic_mIoU = [Metric_mIoU(
        num_classes=self.num_classes,
        use_lidar_mask=False,
        use_image_mask=True, eval_tr=i, with_others=self.with_others) for i in self.eval_threshold_range]
            
        general_IoU = [IoU(use_image_mask=True, eval_tr=i) for i in self.eval_threshold_range]
        
        for index, occ_pred in enumerate(occ_results):
            info = self.data_infos[index]

            occ_path = os.path.join(self.gt_root, info['scene_name'], info['token'], 'labels.npz')
            occ_gt = np.load(occ_path)
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)

            # Add other & other_flat to the ignore class
            if not self.with_others:
                other_mask = (gt_semantics == 0) | (gt_semantics == 12)
                mask_camera = mask_camera & (~other_mask)

            preds = occ_pred['occupancy']
            for i, t in enumerate(self.eval_threshold_range):
                preds_i = preds.copy()
                preds_i[occ_pred['free_space'][i]] = 17
                general_IoU[i].add_batch(preds_i, gt_semantics, mask_lidar, mask_camera)
                semantic_mIoU[i].add_batch(preds_i, gt_semantics, mask_lidar, mask_camera)

        top_mIoU = 0
        top_IoU = 0
        for i, t in enumerate(self.eval_threshold_range):
            print("############################")
            print(f"Eval threshold {t}:")
            general_iou_metric = general_IoU[i]
            iou = general_iou_metric.count_miou()[1][0]
            eval_dict.update({f'IoU_{t}': iou})
            if iou > top_IoU:
                top_IoU = iou

            metric = semantic_mIoU[i]
            miou = metric.count_miou()[2]
            eval_dict.update({f'mIoU_{t}': miou})
            if miou > top_mIoU:
                top_mIoU = miou
            print("############################")
                    
        eval_dict.update({'top_mIoU': top_mIoU})
        eval_dict.update({'top_IoU': top_IoU})

        return eval_dict
    
    def evaluate(self, occ_results, runner=None, show_dir=None, save_dir=None, **eval_kwargs):
        print('\nStarting Evaluation...')
        
        eval_dict = {}
        eval_metrics = self.eval_metrics if 'metric' not in eval_kwargs else eval_kwargs['metric']

        ## Save Occupancy ##
        if save_dir is not None:
            self.save_occupancy(occ_results, save_dir)
            self.save_render_results(occ_results, save_dir)
            self.save_gaussians(occ_results, save_dir)

        ## mIoU for Occ3D v1 ##
        if 'mIoU' in eval_metrics:
            eval_dict = self.evaluate_mIoU(occ_results, eval_dict)

        del occ_results
        gc.collect()
        return eval_dict

    def save_render_results(self, results, out_path):
        if 'rendered_depths' not in results[0]:
            return
        import cv2
        from matplotlib import cm

        _cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                      'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        render_dir = osp.join(out_path, 'render_images')
        for cam in _cam_names:
            mmcv.mkdir_or_exist(osp.join(render_dir, cam))

        magma = cm.get_cmap('magma')

        # standard nuScenes occupancy palette (Occ3D)
        nusc_colors = np.array([
            [0,   0,   0  ],  # 0  undefined
            [255, 120, 50 ],  # 1  barrier
            [255, 192, 203],  # 2  bicycle
            [255, 255, 0  ],  # 3  bus
            [0,   150, 245],  # 4  car
            [0,   255, 255],  # 5  construction vehicle
            [200, 180, 0  ],  # 6  motorcycle
            [255, 0,   0  ],  # 7  pedestrian
            [255, 240, 150],  # 8  traffic cone
            [135, 60,  0  ],  # 9  trailer
            [160, 32,  240],  # 10 truck
            [255, 0,   255],  # 11 driveable surface
            [139, 137, 137],  # 12 other flat
            [75,  0,   75 ],  # 13 sidewalk
            [150, 240, 80 ],  # 14 terrain
            [230, 230, 250],  # 15 manmade
            [0,   175, 0  ],  # 16 vegetation
            [0,   0,   0  ],  # 17 free
        ], dtype=np.uint8)

        for index, output in enumerate(results):
            if 'rendered_depths' not in output:
                continue
            info = self.data_infos[index]
            token = info['token']
            depths = output['rendered_depths']    # [N_cams, H, W] float32
            sems   = output['rendered_semantics'] # [N_cams, H, W] uint8
            h, w = depths.shape[-2:]

            for cam_idx, cam_name in enumerate(_cam_names):
                cam_dir = osp.join(render_dir, cam_name)

                # depth: apply magma colormap over valid (>0) pixels
                d = depths[cam_idx]
                valid = d > 0
                d_norm = np.zeros_like(d)
                if valid.any():
                    d_min, d_max = d[valid].min(), d[valid].max()
                    d_norm[valid] = (d[valid] - d_min) / (d_max - d_min + 1e-6)
                d_color = (magma(d_norm)[..., :3] * 255).astype(np.uint8)
                cv2.imwrite(osp.join(cam_dir, f'{token}_depth.png'),
                            cv2.cvtColor(d_color, cv2.COLOR_RGB2BGR))

                # semantics: nuScenes standard palette
                sem = sems[cam_idx].astype(np.int32)
                sem_color = nusc_colors[np.clip(sem, 0, len(nusc_colors) - 1)]
                cv2.imwrite(osp.join(cam_dir, f'{token}_sem.png'),
                            cv2.cvtColor(sem_color, cv2.COLOR_RGB2BGR))

                # rgb: resize then crop top 140px to match raster_crop_top
                if cam_name in info.get('cams', {}):
                    img = cv2.imread(info['cams'][cam_name]['data_path'])
                    if img is not None:
                        src_h, src_w = img.shape[:2]
                        scale = w / src_w  # 704/1600 = 0.44
                        img_resized = cv2.resize(img, (w, int(src_h * scale)))
                        img_cropped = img_resized[140:140 + h, :]
                        cv2.imwrite(osp.join(cam_dir, f'{token}_rgb.png'), img_cropped)

    def save_occupancy(self, results, out_path):
        mmcv.mkdir_or_exist(out_path)
        all_occs = {}
        for index, output in enumerate(results):
            info = self.data_infos[index]
            scene_name, token = info['scene_name'], info['token']
            occ = output['occupancy']
            occ[output['free_space'][0]] = 17
            # if scene_name not in all_occs.keys():
            #     all_occs[scene_name] = {}
            #     all_fs[scene_name] = {}
            all_occs[token] = occ
            # all_occs[scene_name][token] = occ
            # all_fs[scene_name][token] = output['free_space']

        for token, preds in all_occs.items():
            out_file_occ = osp.join(out_path, f'{token}.npz')
            np.savez(out_file_occ, preds)

    def save_gaussians(self, results, out_path):
        out_path_gauss = os.path.join(out_path, 'gaussians')
        mmcv.mkdir_or_exist(out_path_gauss)
        if 'means' not in results[0].keys():
            # No Gaussians to save
            return
        
        # Save all Gaussians
        all_gaussians = {}
        for index, output in enumerate(results):
            info = self.data_infos[index]
            token = info['token']
            gaussians = {
                'means': output['means'].astype(np.float16),
                'opacity': output['opacity'].astype(np.float16),
                'scale': output['scale'].astype(np.float16),
                'quats': output['quats'].astype(np.float16),
                'label': output['label'].astype(np.uint8),
            }
            all_gaussians[token] = gaussians

        for token, gaussians_per_token in all_gaussians.items():
            np.savez_compressed(osp.join(out_path_gauss, f'{token}.npz'), **gaussians_per_token)
