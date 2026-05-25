"""
scripts/eval_corrupted.py
Evaluasi Point-MAE baseline pada ModelNet40 clean + 3 corruption x 5 severity.

Usage (dari dalam folder Point-MAE/):
    python ../pointmae-baseline/scripts/eval_corrupted.py \
        --ckpt ./modelnet_1k.pth \
        --dat  ./data/ModelNet/modelnet40_normal_resampled/modelnet40_test_1024pts_fps_clean.dat
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ─────────────────────────────────────────────
# 1. ARGS
# ─────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',        type=str, required=True,
                        help='path to modelnet_1k.pth')
    parser.add_argument('--dat',         type=str, required=True,
                        help='path to modelnet40_test_1024pts_fps_clean.dat')
    parser.add_argument('--npoints',     type=int, default=1024)
    parser.add_argument('--batch_size',  type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--out',         type=str, default='results_baseline.txt',
                        help='save hasil ke file txt')
    return parser.parse_args()


# ─────────────────────────────────────────────
# 2. CORRUPTION FUNCTIONS
# ─────────────────────────────────────────────

def pc_normalize(pc):
    pc = pc - np.mean(pc, axis=0)
    return pc / (np.max(np.linalg.norm(pc, axis=1)) + 1e-8)


def corrupt_sparse(pts, severity):
    """Distance-decay dropout — simulasi atenuasi LiDAR."""
    alpha = [0.20, 0.40, 0.60, 0.80, 0.95][severity - 1]
    tau   = [1.00, 0.80, 0.60, 0.40, 0.30][severity - 1]
    dist  = np.linalg.norm(pts, axis=1)
    p_drop = alpha * (1 - np.exp(-dist ** 2 / tau))
    keep  = np.random.rand(len(pts)) > p_drop
    kept  = pts[keep] if keep.sum() >= 64 else pts
    idx   = np.random.choice(len(kept), size=len(pts), replace=True)
    return kept[idx]


def corrupt_line(pts, sev):
    gamma = [0.05, 0.10, 0.20, 0.30, 0.45][sev-1]
    K = 32
    
    # Konversi ke spherical, bin elevation -> ring assignment
    r = np.linalg.norm(pts, axis=1, keepdims=True).clip(1e-6)
    theta = np.arcsin(np.clip(pts[:,2:3]/r, -1, 1))
    bins = np.linspace(theta.min(), theta.max()+1e-6, K+1)
    ring = np.clip(np.digitize(theta[:,0], bins)-1, 0, K-1)
    
    # Setiap ring di-drop secara independen ~ Bernoulli(gamma)
    drop_mask = np.random.rand(K) < gamma
    dropped_rings = np.where(drop_mask)[0]
    
    # Fallback: kalau semua ring ke-drop, paksa keep minimal 1 ring
    if len(dropped_rings) == K:
        dropped_rings = dropped_rings[:-1]
    
    keep = ~np.isin(ring, dropped_rings)
    kept = pts[keep] if keep.sum() >= 64 else pts
    return kept[np.random.choice(len(kept), len(pts), replace=True)]


def corrupt_occlusion(pts, severity):
    """Spatial cluster removal — simulasi shadow region."""
    rho_mean = [0.10, 0.15, 0.20, 0.25, 0.30][severity - 1]
    kept = pts.copy()
    for _ in range(2):
        if len(kept) < 64:
            break
        u    = kept[np.random.randint(len(kept))]
        rho  = np.clip(np.random.normal(rho_mean, 0.02), 0.05, 0.40)
        mask = np.linalg.norm(kept - u, axis=1) > rho
        if mask.sum() >= 64:
            kept = kept[mask]
    idx = np.random.choice(len(kept), size=len(pts), replace=True)
    return kept[idx]

CORRUPTION_FNS = {
    'sparse':    corrupt_sparse,
    'line':      corrupt_line,
    'occlusion': corrupt_occlusion,
}


# ─────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────

class ModelNet40Clean(Dataset):
    """
    Load dari .dat yang sudah pre-computed (1024 pts, normalized).
    Format: (pts_array (N,1024,3), labels_array (N,))
    Apply corruption on-the-fly.
    """
    def __init__(self, dat_path, corruption=None, severity=0):
        self.corruption = corruption
        self.severity   = severity

        with open(dat_path, 'rb') as f:
            pts, lbl = pickle.load(f)

        self.points = np.array(pts, dtype=np.float32)        # (N, 1024, 3)
        self.labels = np.array(lbl, dtype=np.int64).squeeze() # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pts   = self.points[idx].copy()   # (1024, 3)
        label = int(self.labels[idx])

        if self.corruption is not None:
            fn  = CORRUPTION_FNS[self.corruption]
            pts = fn(pts, self.severity).astype(np.float32)
            pts = pc_normalize(pts).astype(np.float32)

        return torch.from_numpy(pts).float(), label


# ─────────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────────

def build_model(ckpt_path):
    try:
        from models.Point_MAE import PointTransformer
        from utils.config import EasyDict
    except ImportError as e:
        print(f'[ERROR] {e}')
        print('Pastikan script dijalankan dari dalam folder Point-MAE/')
        print('Dan sudah apply patches/ terlebih dahulu.')
        sys.exit(1)

    cfg = EasyDict()
    cfg.trans_dim      = 384
    cfg.depth          = 12
    cfg.drop_path_rate = 0.1
    cfg.cls_dim        = 40
    cfg.num_heads      = 6
    cfg.group_size     = 32
    cfg.num_group      = 64
    cfg.encoder_dims   = 384

    model = PointTransformer(cfg)
    model.load_model_from_ckpt(ckpt_path)
    model = nn.DataParallel(model).cuda()
    model.eval()
    return model


# ─────────────────────────────────────────────
# 5. EVAL
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader):
    correct = total = 0
    for pts, labels in tqdm(loader, leave=False):
        pred    = model(pts.cuda()).argmax(-1)
        correct += (pred == labels.cuda()).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    args = get_args()

    print(f'Loading checkpoint : {args.ckpt}')
    model = build_model(args.ckpt)
    print(f'Loading data       : {args.dat}\n')

    results  = {}
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    # ── Clean ──
    log('Evaluating: CLEAN')
    loader = DataLoader(
        ModelNet40Clean(args.dat),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False
    )
    oa = evaluate(model, loader)
    results['clean'] = oa
    log(f'  OA: {oa:.2f}%\n')

    # ── Corrupted ──
    for corruption in ['sparse', 'line', 'occlusion']:
        results[corruption] = {}
        for severity in range(1, 6):
            log(f'Evaluating: {corruption.upper()} severity={severity}')
            loader = DataLoader(
                ModelNet40Clean(args.dat, corruption=corruption, severity=severity),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, drop_last=False
            )
            oa = evaluate(model, loader)
            results[corruption][severity] = oa
            log(f'  OA: {oa:.2f}%')

        robust = np.mean([results[corruption][s] for s in range(1, 6)])
        results[corruption]['robust'] = robust
        log(f'  Robust Acc ({corruption}): {robust:.2f}%\n')

    # ── Tabel ──
    sep = '=' * 68
    log('\n' + sep)
    log(f'{"":14}  S1       S2       S3       S4       S5      Robust')
    log(sep)
    log(f'{"Clean":14}  {results["clean"]:.2f}%')
    for c in ['sparse', 'line', 'occlusion']:
        r = results[c]
        log(f'{c:14}  '
            f'{r[1]:.2f}%   {r[2]:.2f}%   {r[3]:.2f}%   '
            f'{r[4]:.2f}%   {r[5]:.2f}%   {r["robust"]:.2f}%')
    log(sep)

    # ── mCE ──
    # Reference: DGCNN Robust Acc dari Ren et al. (2022) — update sesuai paper
    dgcnn_robust = {'sparse': 75.0, 'line': 70.0, 'occlusion': 72.0}
    log('\nmCE (relative to DGCNN baseline from Ren et al. 2022):')
    mce_list = []
    for c in ['sparse', 'line', 'occlusion']:
        ce_model = 1 - results[c]['robust'] / 100
        ce_dgcnn = 1 - dgcnn_robust[c] / 100
        mce      = ce_model / ce_dgcnn
        mce_list.append(mce)
        log(f'  CE_{c:12}: {mce:.4f}')
    log(f'  mCE overall    : {np.mean(mce_list):.4f}')
    log('\n[!] Update dgcnn_robust values sesuai Tabel 1 Ren et al. (2022)')

    # ── Save ──
    with open(args.out, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f'\nResults saved to: {args.out}')


if __name__ == '__main__':
    main()
