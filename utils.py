"""Utility functions for loading images and masks.
Designed to be imported, so names won't conflict with notebook variables.
"""
import os
from pathlib import Path
import numpy as np
import cv2

try:
    import nibabel as nib
    HAVE_NIB = True
except Exception:
    HAVE_NIB = False

def _read_image(path, size=(128,128), grayscale=False):
    path = str(path)
    ext = Path(path).suffix.lower()
    if ext in ('.nii', '.gz') and HAVE_NIB:
        img = nib.load(path).get_fdata()
        # take middle slice if 3D
        if img.ndim == 3:
            img = img[:, :, img.shape[2]//2]
        img = cv2.resize(img.astype('float32'), size)
        if grayscale:
            img = img[..., None]
        return img
    # else read via cv2
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Can't read image: {path}")
    if not grayscale and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, size)
    if grayscale:
        img = img[..., None]
    return img.astype('float32')

def normalize_img(x):
    # scale to [0,1]
    return x / 255.0

def load_image_mask_pairs(image_dir, mask_dir, size=(128,128), as_gray=True, max_files=None):
    """Load image/mask pairs from two directories. Filenames must match.
    Returns numpy arrays (X, y).
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    images = sorted([p for p in image_dir.iterdir() if p.is_file()])
    if max_files:
        images = images[:max_files]
    X = []
    y = []
    for img_path in images:
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            # try common image extensions
            stem = img_path.stem
            found = None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                p = mask_dir / (stem + ext)
                if p.exists():
                    found = p; break
            if not found:
                # skip if no matching mask
                continue
            mask_path = found
        img = _read_image(img_path, size=size, grayscale=as_gray)
        m = _read_image(mask_path, size=size, grayscale=True)
        X.append(normalize_img(img))
        # ensure mask is binary {0,1}
        m = (m > 127).astype('float32')
        y.append(m)
    if len(X) == 0:
        return None, None
    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)
    return X, y

def train_val_test_split(X, y, val_ratio=0.1, test_ratio=0.1, shuffle=True, seed=42):
    import numpy as _np
    n = len(X)
    idx = _np.arange(n)
    if shuffle:
        rng = _np.random.RandomState(seed)
        rng.shuffle(idx)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test+n_val]
    train_idx = idx[n_test+n_val:]
    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])
