import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

PALETTE = np.array([
    [255,120,50],[255,192,203],[255,255,0],[0,150,245],[0,255,255],
    [255,127,0],[255,0,0],[255,240,150],[135,60,0],[160,32,240],
    [255,0,255],[139,137,137],[75,0,75],[150,240,80],[230,230,250],
    [0,175,0],
], dtype=np.float32) / 255

FREE          = 17
NUSC_DATAROOT = "data/nuscenes"
GT_ROOT       = "data/gts"
PRED_DIR      = "occ/gaussianflowocc"
OUT_DIR_PRED  = "occ/gaussianflowocc/pred"
OUT_DIR_GT    = "occ/gaussianflowocc/gt"

def occ_to_bev(occ, free):
    bev = np.full(occ.shape[:2], free, dtype=np.uint8)
    for z in range(occ.shape[2]):
        layer = occ[:, :, z]
        m = layer != free
        bev[m] = layer[m]
    return bev


def bev_to_img(bev, free):
    img = np.ones((bev.shape[0], bev.shape[1], 3))
    occupied = bev != free
    img[occupied] = PALETTE[(bev[occupied] - 1) % len(PALETTE)]
    return img


def visualize(nusc, token, pred_npz_path):
    pred = np.load(pred_npz_path)["arr_0"]

    sample = nusc.get("sample", token)
    scene_name = nusc.get("scene", sample["scene_token"])["name"]
    gt_path = f"{GT_ROOT}/{scene_name}/{token}/labels.npz"

    gt_raw = np.load(gt_path)
    gt = gt_raw[gt_raw.files[0]]

    gt_free = 0 if 0 in np.unique(gt) and 17 not in np.unique(gt) else 17

    pred_img = bev_to_img(occ_to_bev(pred, FREE), FREE)
    gt_img   = bev_to_img(occ_to_bev(gt, gt_free), gt_free)

    plt.imsave(f"{OUT_DIR_PRED}/{token}.png", pred_img)
    plt.imsave(f"{OUT_DIR_GT}/{token}.png",   gt_img)


if __name__ == "__main__":
    os.makedirs(OUT_DIR_PRED, exist_ok=True)
    os.makedirs(OUT_DIR_GT,   exist_ok=True)

    nusc = NuScenes(dataroot=NUSC_DATAROOT, version="v1.0-trainval", verbose=False)

    npz_files = [f for f in os.listdir(PRED_DIR) if f.endswith(".npz")]
    print(f"Found {len(npz_files)} npz files")

    for fname in tqdm(npz_files):
        token = fname.replace(".npz", "")
        try:
            visualize(nusc, token, os.path.join(PRED_DIR, fname))
        except Exception as e:
            print(f"[SKIP] {token}: {e}")