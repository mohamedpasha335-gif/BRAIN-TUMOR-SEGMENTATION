# Segmentation Project - Organized Files

This folder contains modularized code extracted from the notebook.

Files:
- utils.py: dataset loading and preprocessing
- losses.py: dice loss and metrics
- unet.py: U-Net model builder
- classification.py: example EfficientNet classifier builder
- train_segmentation.py: training script for U-Net
- streamlit_app.py: simple Streamlit app for inference
- requirements.txt: packages to install

Quick start:
1. Create virtual env and install requirements:
    python -m venv venv
    source venv/bin/activate  # (or venv\Scripts\activate on Windows)
    pip install -r requirements.txt
2. Train (example):
    python train_segmentation.py --images path/to/images --masks path/to/masks --epochs 10
3. Run Streamlit inference:
    streamlit run streamlit_app.py



https://drive.google.com/file/d/1Kf1w63FbYfIR7adoVMszII_kS66Rxo6b/view?usp=drive_link(LINK SEGMENTATION PRO )
