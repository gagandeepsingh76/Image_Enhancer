# MIRNet Low-Light Image Enhancement

This project implements the MIRNet model for low-light image enhancement using TensorFlow and Keras.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Download the LOL dataset
2. Train the MIRNet model
3. Save the best model as 'best_model.h5'
4. Create a 'results' directory with enhanced images
5. Show comparisons between original, PIL autocontrast, and MIRNet enhanced images

## Model Architecture

The MIRNet model uses a multi-scale residual network architecture with:
- Multi-scale feature extraction
- Parallel attention mechanisms
- Residual learning
- Skip connections

## Training

The model is trained using:
- Charbonnier loss function
- Adam optimizer
- Peak Signal-to-Noise Ratio (PSNR) metric
- Early stopping to prevent overfitting

## Results

Results are saved in the 'results' directory:
- Enhanced images
- Comparison plots
- Training history plots 