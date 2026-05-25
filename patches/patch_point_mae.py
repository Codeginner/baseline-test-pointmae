"""
patches/patch_point_mae.py
Fix models/Point_MAE.py agar tidak bergantung pada knn_cuda.
Ganti dengan pure PyTorch KNN fallback.

Usage (dari dalam folder Point-MAE/):
    python ../pointmae-baseline/patches/patch_point_mae.py
"""

import os
import sys

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Point-MAE', 'models', 'Point_MAE.py')

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join('models', 'Point_MAE.py')

if not os.path.exists(MODEL_PATH):
    print('[ERROR] Point_MAE.py tidak ditemukan. Jalankan dari dalam folder Point-MAE/')
    sys.exit(1)

print(f'Patching: {MODEL_PATH}')

with open(MODEL_PATH, 'r') as f:
    content = f.read()

OLD_IMPORT = 'from knn_cuda import KNN'

NEW_IMPORT = '''try:
    from knn_cuda import KNN
except ImportError:
    class KNN:
        """Pure PyTorch fallback untuk knn_cuda."""
        def __init__(self, k, transpose_mode=True):
            self.k = k
            self.transpose_mode = transpose_mode

        def __call__(self, ref, query):
            # ref, query: (B, 3, N) kalau transpose_mode=True
            import torch
            if self.transpose_mode:
                ref_t   = ref.permute(0, 2, 1).float()    # (B, N, 3)
                query_t = query.permute(0, 2, 1).float()   # (B, M, 3)
            else:
                ref_t, query_t = ref.float(), query.float()

            # cdist: (B, M, N)
            dist = torch.cdist(query_t, ref_t)
            val, idx = dist.topk(self.k, dim=-1, largest=False)  # (B, M, k)

            if self.transpose_mode:
                val = val.permute(0, 2, 1)   # (B, k, M)
                idx = idx.permute(0, 2, 1)   # (B, k, M)

            return val, idx'''

if OLD_IMPORT in content:
    content = content.replace(OLD_IMPORT, NEW_IMPORT)
    print('  [OK] Patched knn_cuda import')
else:
    print('  [SKIP] knn_cuda import tidak ditemukan atau sudah di-patch')

with open(MODEL_PATH, 'w') as f:
    f.write(content)

print('Done. Point_MAE.py patched successfully.')
