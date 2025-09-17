# Cell Continuous Value GAN

This project implements a Generative Adversarial Network (GAN) for generating realistic cell continuous value maps from training data containing ~100 cell images with continuous probability values.

## Features

- **Continuous Value Generation**: The GAN generates 32-bit continuous cell values (range [-1,1]) for improved training stability
- **WGAN-GP Architecture**: Uses Wasserstein GAN with Gradient Penalty for stable training
- **Flexible Input Handling**: Supports various input formats (uint8, uint16, float32) and normalizes them appropriately
- **Comprehensive Logging**: Detailed training progress with loss tracking and sample generation

## Model Architecture

### Generator
- **Input**: Random noise vector (latent dimension: 100)
- **Output**: 256x256 single-channel continuous value maps (range [-1,1])
- **Activation**: Tanh output activation for stable training in [-1,1] range
- **Architecture**: Deep Convolutional Transpose layers with batch normalization

### Discriminator
- **Input**: 256x256 single-channel continuous value maps (range [-1,1])
- **Output**: Scalar score (no sigmoid for WGAN-GP)
- **Architecture**: Convolutional layers with batch normalization and LeakyReLU

## Key Changes for Continuous Values

1. **Data Loading**: Modified to handle any continuous input range and normalize to [-1,1] for training
2. **Generator Output**: Uses Tanh activation for [-1,1] output range (better training stability)
3. **Data Normalization**: Input values normalized to [-1,1] range throughout pipeline
4. **Visualization**: Updated to handle continuous values with proper range conversion for display

## Usage

```bash
python main.py --data_dir /path/to/probability/masks --output_dir /path/to/output
```

## Input Data Format

The model accepts flexible input formats:
- **32-bit float continuous values** (any range - automatically normalized)
- **16-bit integer values** (automatically normalized to appropriate range)
- **8-bit integer values** (automatically normalized to appropriate range)

All inputs are automatically normalized to the [-1,1] range for optimal training stability.

## Output

- Generated continuous value maps as TIFF files (both 8-bit and 32-bit float formats)
- Training progress comparisons (real vs generated)
- Model checkpoints every 50 epochs
- Loss curves and training metrics

## Output Formats

- **8-bit TIFF**: Converted to [0,255] range for standard image viewing
- **32-bit Float TIFF**: Raw [-1,1] values for full precision
- **Normalized 32-bit Float TIFF**: Converted to [0,1] range for easier interpretation