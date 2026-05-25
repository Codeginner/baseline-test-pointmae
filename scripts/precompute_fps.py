"""
scripts/precompute_fps.py
Pre-compute FPS dari 8192 -> 1024 pts untuk ModelNet40 test set.
Hasilnya disimpan ke .dat baru agar eval lebih cepat.

Usage:
    python scripts/precompute_fps.py \
        --dat ./data/ModelNet/modelnet40_normal_resampled/modelnet40_test_8192pts_fps.dat \
        --out ./data/ModelNet/modelnet40_normal_resampled/modelnet40_test_1024pts_fps_clean.dat \
        --npoints 1024
"""

import argparse
import pickle
import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dat',     type=str, required=True,  help='path ke .dat 8192 pts')
    parser.add_argument('--out',     type=str, required=True,  help='output path .dat 1024 pts')
    parser.add_argument('--npoints', type=int, default=1024,   help='target jumlah titik')
    return parser.parse_args()


def pc_normalize(pc):
    pc = pc - np.mean(pc, axis=0)
    m  = np.max(np.linalg.norm(pc, axis=1))
    return pc / (m + 1e-8)


def fps_numpy(pts, n):
    """Farthest Point Sampling — pure numpy."""
    N   = len(pts)
    sel = np.zeros(n, dtype=int)
    dist = np.full(N, 1e10)
    far  = np.random.randint(N)
    for i in range(n):
        sel[i] = far
        d    = np.sum((pts - pts[far]) ** 2, axis=-1)
        dist = np.minimum(dist, d)
        far  = np.argmax(dist)
    return pts[sel]


def main():
    args = get_args()

    print(f'Loading: {args.dat}')
    with open(args.dat, 'rb') as f:
        raw_pts, raw_lbl = pickle.load(f)

    raw_pts = np.array(raw_pts, dtype=np.float32)        # (2468, 8192, 6)
    raw_lbl = np.array(raw_lbl, dtype=np.int64).squeeze() # (2468,)
    print(f'Loaded {len(raw_lbl)} samples — pts shape: {raw_pts.shape}')

    pts_out = []
    for i in tqdm(range(len(raw_lbl)), desc=f'FPS {raw_pts.shape[1]}->{args.npoints}'):
        pts = raw_pts[i, :, :3].copy()          # (8192, 3) — xyz only
        pts = fps_numpy(pts, args.npoints)       # (1024, 3)
        pts = pc_normalize(pts).astype(np.float32)
        pts_out.append(pts)

    pts_out = np.array(pts_out, dtype=np.float32)  # (2468, 1024, 3)
    print(f'FPS done — output shape: {pts_out.shape}')

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump((pts_out, raw_lbl), f)

    print(f'Saved to: {args.out}')


if __name__ == '__main__':
    main()
