# import matplotlib
# matplotlib.use("Agg")

# import os
# import numpy as np
# import matplotlib.pyplot as plt

# PALETTE = np.array([
#     [255,120,50],[255,192,203],[255,255,0],[0,150,245],[0,255,255],
#     [255,127,0],[255,0,0],[255,240,150],[135,60,0],[160,32,240],
#     [255,0,255],[139,137,137],[75,0,75],[150,240,80],[230,230,250],
#     [0,175,0],
# ], dtype=np.float32) / 255

# occ = np.load("occ/gaussianflowocc/0a0d6b8c2e884134a3b48df43d54c36a.npz")["arr_0"]

# FREE = 17
# bev = np.full(occ.shape[:2], FREE, dtype=np.uint8)
# for z in range(occ.shape[2]):
#     layer = occ[:, :, z]
#     m = layer != FREE
#     bev[m] = layer[m]

# img = np.ones((occ.shape[0], occ.shape[1], 3))
# occupied = bev != FREE
# img[occupied] = PALETTE[(bev[occupied] - 1) % len(PALETTE)]

# os.makedirs("shen_scripts", exist_ok=True)
# plt.imsave("shen_scripts/bev.png", img)
# print("DONE")