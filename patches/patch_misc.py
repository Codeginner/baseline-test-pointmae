"""
patches/patch_misc.py
Fix utils/misc.py agar tidak bergantung pada pointnet2_ops.
Ganti dengan pure PyTorch FPS fallback.

Usage (dari dalam folder Point-MAE/):
    python ../pointmae-baseline/patches/patch_misc.py
"""

import os
import sys

MISC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Point-MAE', 'utils', 'misc.py')

# Kalau dijalankan dari dalam Point-MAE/
if not os.path.exists(MISC_PATH):
    MISC_PATH = os.path.join('utils', 'misc.py')

if not os.path.exists(MISC_PATH):
    print(f'[ERROR] misc.py tidak ditemukan. Jalankan dari dalam folder Point-MAE/')
    sys.exit(1)

print(f'Patching: {MISC_PATH}')

with open(MISC_PATH, 'r') as f:
    lines = f.readlines()

# Cari line yang import pointnet2_ops
import_line = None
fps_line1  = None
fps_line2  = None

for i, line in enumerate(lines):
    if 'from pointnet2_ops import pointnet2_utils' in line and 'try' not in line:
        import_line = i
    if 'pointnet2_utils.furthest_point_sample' in line:
        fps_line1 = i
    if 'pointnet2_utils.gather_operation' in line:
        fps_line2 = i

print(f'  import line  : {import_line}')
print(f'  fps_line1    : {fps_line1}')
print(f'  fps_line2    : {fps_line2}')

# Patch import
if import_line is not None and 'try:' not in lines[max(0, import_line-1)]:
    lines[import_line] = (
        'try:\n'
        '    from pointnet2_ops import pointnet2_utils\n'
        'except ImportError:\n'
        '    pointnet2_utils = None\n'
    )
    print('  [OK] Patched import')

# Patch fps function body
if fps_line1 is not None and fps_line2 is not None:
    lines[fps_line1] = (
        '    if pointnet2_utils is not None:\n'
        '        fps_idx = pointnet2_utils.furthest_point_sample(data, number)\n'
        '        fps_data = pointnet2_utils.gather_operation(\n'
        '            data.transpose(1, 2).contiguous(), fps_idx\n'
        '        ).transpose(1,2).contiguous()\n'
        '    else:\n'
        '        import torch\n'
        '        B, N, C = data.shape\n'
        '        fps_idx = torch.zeros(B, number, dtype=torch.long, device=data.device)\n'
        '        dist = torch.full((B, N), 1e10, device=data.device)\n'
        '        far = torch.randint(0, N, (B,), device=data.device)\n'
        '        for i in range(number):\n'
        '            fps_idx[:, i] = far\n'
        '            centroid = data[torch.arange(B, device=data.device), far].unsqueeze(1)\n'
        '            d = torch.sum((data - centroid) ** 2, dim=-1)\n'
        '            dist = torch.minimum(dist, d)\n'
        '            far = torch.argmax(dist, dim=-1)\n'
        '        fps_data = data[\n'
        '            torch.arange(B, device=data.device).unsqueeze(1), fps_idx\n'
        '        ]\n'
    )
    lines[fps_line2] = ''
    print('  [OK] Patched fps function')

with open(MISC_PATH, 'w') as f:
    f.writelines(lines)

print('Done. misc.py patched successfully.')
