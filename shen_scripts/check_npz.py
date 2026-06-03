import numpy as np

d = np.load("occ/gaussianflowocc/0a0d6b8c2e884134a3b48df43d54c36a.npz")

print("Keys:", d.files)
for k in d.files:
    arr = d[k]
    print(f"\n{k}: shape={arr.shape}, dtype={arr.dtype}")
    print("  unique values:", np.unique(arr))