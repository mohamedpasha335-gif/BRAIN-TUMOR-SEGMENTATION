"""Train script for segmentation model.
Usage example:
    python train_segmentation.py --images path/to/images --masks path/to/masks --epochs 10
"""
import argparse, os
from pathlib import Path
import numpy as np
import tensorflow as tf

from utils import load_image_mask_pairs, train_val_test_split
from unet import create_unet
import losses

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images', required=True)
    p.add_argument('--masks', required=True)
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--out', default='unet_segmentation.h5')
    return p.parse_args()

def main():
    args = parse_args()
    X, y = load_image_mask_pairs(args.images, args.masks, size=(args.size,args.size))
    if X is None:
        raise SystemExit('No data found. Check paths.')
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, val_ratio=0.1, test_ratio=0.1)

    model = create_unet(input_size=(args.size,args.size, X.shape[-1]))
    model.compile(optimizer='adam', loss=losses.dice_loss, metrics=['accuracy', losses.dice_coef, losses.iou_metric])

    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=args.batch)
    model.save(args.out)
    print('Model saved to', args.out)

if __name__ == '__main__':
    main()
