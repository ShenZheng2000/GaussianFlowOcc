# # TODO: rewrite to make this looks better! 
# # TODO: rewrite to accept batch processing! 

# import matplotlib
# matplotlib.use("Agg")

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from nuscenes.nuscenes import NuScenes

# PALETTE = np.array([
#     [255,120,50],[255,192,203],[255,255,0],[0,150,245],[0,255,255],
#     [255,127,0],[255,0,0],[255,240,150],[135,60,0],[160,32,240],
#     [255,0,255],[139,137,137],[75,0,75],[150,240,80],[230,230,250],
#     [0,175,0],
# ], dtype=np.float32) / 255

# FREE = 17
# NUSC_DATAROOT = "/scratch/shenzhen/Datasets/nuscenes"
# GT_ROOT = "/scratch/shenzhen/Datasets/Occ3D-nuScenes/voxel04/gts"
# OUT_DIR = "shen_scripts"


# def occ_to_bev(occ, free):
#     bev = np.full(occ.shape[:2], free, dtype=np.uint8)
#     for z in range(occ.shape[2]):
#         layer = occ[:, :, z]
#         m = layer != free
#         bev[m] = layer[m]
#     return bev


# def bev_to_img(bev, free):
#     img = np.ones((bev.shape[0], bev.shape[1], 3))
#     occupied = bev != free
#     img[occupied] = PALETTE[(bev[occupied] - 1) % len(PALETTE)]
#     return img


# # def get_front_cam_image(nusc, token):
# #     sample = nusc.get("sample", token)
# #     cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
# #     return plt.imread(f"{NUSC_DATAROOT}/{cam['filename']}")


# def visualize(token, pred_npz_path):
#     nusc = NuScenes(dataroot=NUSC_DATAROOT, version="v1.0-trainval", verbose=False)

#     pred = np.load(pred_npz_path)["arr_0"]
#     print(f"[pred] keys: {np.load(pred_npz_path).files}")
#     print(f"[pred] shape={pred.shape}, dtype={pred.dtype}, unique={np.unique(pred)}")

#     # gt_path = f"{GT_ROOT}/{token}/labels.npz"
#     sample = nusc.get("sample", token)
#     scene_name = nusc.get("scene", sample["scene_token"])["name"]
#     print(f"[gt]   scene={scene_name}")
#     gt_path = f"{GT_ROOT}/{scene_name}/{token}/labels.npz"

#     gt_raw = np.load(gt_path)
#     print(f"[gt]   keys: {gt_raw.files}")
#     gt_key = gt_raw.files[0]
#     gt = gt_raw[gt_key]
#     print(f"[gt]   key used='{gt_key}', shape={gt.shape}, dtype={gt.dtype}, unique={np.unique(gt)}")

#     gt_free = 0 if 0 in np.unique(gt) and 17 not in np.unique(gt) else 17
#     print(f"[gt]   using free={gt_free}")

#     # cam_img = get_front_cam_image(nusc, token)
#     # print(f"[cam]  image shape={cam_img.shape}")

#     pred_img = bev_to_img(occ_to_bev(pred, FREE), FREE)
#     gt_img   = bev_to_img(occ_to_bev(gt, gt_free), gt_free)

#     # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     # axes[0].imshow(cam_img);  axes[0].set_title("CAM_FRONT", fontsize=18);  axes[0].axis("off")
#     # axes[1].imshow(pred_img); axes[1].set_title("Pred BEV", fontsize=18);   axes[1].axis("off")
#     # axes[2].imshow(gt_img);   axes[2].set_title("GT BEV", fontsize=18);     axes[2].axis("off")

#     from matplotlib.gridspec import GridSpec

#     fig = plt.figure(figsize=(26, 6))
#     # gs = GridSpec(1, 3, figure=fig)
#     # gs = GridSpec(1, 3, figure=fig, wspace=0.05)
#     gs = GridSpec(1, 3, figure=fig, wspace=0.02, width_ratios=[2, 1, 1])

#     # gs_left = gs[0, 0].subgridspec(2, 3, hspace=0.05, wspace=0.05)
#     gs_left = gs[0, 0].subgridspec(2, 3, hspace=0.0, wspace=0.0)

#     # fig.text(0.17, 1.01, "Input Images", ha="center", fontsize=18)

#     cam_order = [
#         ["CAM_FRONT_LEFT", "CAM_FRONT",       "CAM_FRONT_RIGHT"],
#         ["CAM_BACK_LEFT",  "CAM_BACK",        "CAM_BACK_RIGHT" ],
#     ]
#     for r, row in enumerate(cam_order):
#         for c, cam_name in enumerate(row):
#             ax = fig.add_subplot(gs_left[r, c])
#             cam_data = nusc.get("sample_data", sample["data"][cam_name])
#             cimg = plt.imread(f"{NUSC_DATAROOT}/{cam_data['filename']}")
#             ax.imshow(cimg)
#             # ax.set_title(cam_name.replace("CAM_", ""), fontsize=8)
#             ax.axis("off")

#     ax_pred = fig.add_subplot(gs[0, 1])
#     ax_pred.imshow(pred_img); ax_pred.set_title("Pred BEV", fontsize=18); ax_pred.axis("off")

#     ax_gt = fig.add_subplot(gs[0, 2])
#     ax_gt.imshow(gt_img); ax_gt.set_title("GT BEV", fontsize=18); ax_gt.axis("off")

#     # fig.tight_layout()
#     # fig.tight_layout(rect=[0, 0, 1, 0.95])
#     fig.tight_layout()
#     fig.subplots_adjust(top=0.85, left=0.01, right=0.99)
#     fig.text(0.17, 0.95, "Input Images", ha="center", fontsize=18)

#     os.makedirs(OUT_DIR, exist_ok=True)
#     out_path = f"{OUT_DIR}/{token}_compare.png"
#     plt.savefig(out_path, dpi=150)
#     plt.close()
#     print(f"Saved → {out_path}")


# if __name__ == "__main__":
#     TOKEN = "0a0d6b8c2e884134a3b48df43d54c36a"
#     visualize(TOKEN, f"occ/gaussianflowocc/{TOKEN}.npz")