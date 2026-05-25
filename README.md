# Point-MAE Baseline Evaluation on Corrupted ModelNet40

Repo ini berisi script untuk mengevaluasi Point-MAE baseline pada ModelNet40 bersih dan terkorupsi (sparse, line, occlusion), sebagai bagian dari penelitian **Rekonstruksi 3D Point Cloud menggunakan Masking Stokastik**.

## Struktur Repo

```
pointmae-baseline/
├── scripts/
│   ├── precompute_fps.py       # Pre-compute FPS 8192 -> 1024 pts
│   └── eval_corrupted.py       # Evaluasi baseline pada clean + corrupted
├── patches/
│   ├── patch_misc.py           # Fix pointnet2_ops di utils/misc.py
│   └── patch_point_mae.py      # Fix knn_cuda di models/Point_MAE.py
├── requirements.txt
└── README.md
```

## Setup di Kaggle (T4x2)

### 1. Clone Point-MAE dan repo ini

```bash
git clone https://github.com/Pang-Yatian/Point-MAE.git
git clone https://github.com/<your-username>/pointmae-baseline.git
```

### 2. Download checkpoint resmi Point-MAE

```bash
wget https://github.com/Pang-Yatian/Point-MAE/releases/download/main/modelnet_1k.pth \
    -O Point-MAE/modelnet_1k.pth
```

### 3. Download dataset ModelNet40 (8192 pts)

Download `modelnet40_test_8192pts_fps.dat` dari Google Drive dan taruh di:
```
Point-MAE/data/ModelNet/modelnet40_normal_resampled/modelnet40_test_8192pts_fps.dat
```

### 4. Apply patches (fix dependency CUDA 12.8)

```bash
cd Point-MAE
python ../pointmae-baseline/patches/patch_misc.py
python ../pointmae-baseline/patches/patch_point_mae.py
```

### 5. Pre-compute FPS (sekali saja, ~20-30 menit)

```bash
python ../pointmae-baseline/scripts/precompute_fps.py \
    --dat ./data/ModelNet/modelnet40_normal_resampled/modelnet40_test_8192pts_fps.dat \
    --out ./data/ModelNet/modelnet40_normal_resampled/modelnet40_test_1024pts_fps_clean.dat
```

### 6. Jalankan evaluasi

```bash
cd Point-MAE
python ../pointmae-baseline/scripts/eval_corrupted.py \
    --ckpt ./modelnet_1k.pth \
    --dat ./data/ModelNet/modelnet40_normal_resampled/modelnet40_test_1024pts_fps_clean.dat
```

## Output

```
=================================================================
             S1       S2       S3       S4       S5     Robust
=================================================================
Clean         XX.XX%
sparse        XX.XX%   XX.XX%   XX.XX%   XX.XX%   XX.XX%   XX.XX%
line          XX.XX%   XX.XX%   XX.XX%   XX.XX%   XX.XX%   XX.XX%
occlusion     XX.XX%   XX.XX%   XX.XX%   XX.XX%   XX.XX%   XX.XX%
=================================================================
```

## Corruption Definitions

| Type | Simulasi | Parameter |
|---|---|---|
| Sparse | Distance-decay dropout | α=[0.2,0.4,0.6,0.8,0.95], τ=[1.0,0.8,0.6,0.4,0.3] |
| Line | Ring-beam dropout | n_drop=[1,2,3,4,5] rings dari K=32 |
| Occlusion | Spatial cluster removal | ρ_mean=[0.10,0.15,0.20,0.25,0.30] |

## Environment

- Python 3.12
- PyTorch 2.10 + CUDA 12.8
- Kaggle T4x2

## Referensi

- Pang et al. (2022). Masked Autoencoders for Point Cloud Self-supervised Learning. ECCV 2022.
- Sun et al. (2022). Benchmarking Robustness of 3D Point Cloud Recognition against Common Corruptions.
- Ren et al. (2022). Benchmarking and Analyzing Point Cloud Classification under Corruptions. ICML 2022.
