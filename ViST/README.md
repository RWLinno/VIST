# ViST: Vision-enhanced Spatio-Temporal Forecasting

### Quick Start
```bash
python experiments/train.py -c VIST/train_SD.py -g 0
```


## Introduction

ViST (Vision-enhanced Spatio-Temporal forecasting) is an innovative framework that adapts the concept of vision-enhanced time series forecasting for spatio-temporal data. The model transforms raw spatio-temporal data into multiple visual representations (segmentation maps, Gramian Angular Fields, Recurrence Plots, and spatial adjacency images), which are then processed alongside frequency domain features and text descriptions to generate accurate forecasts.

The core idea is to leverage the strengths of computer vision techniques to capture complex patterns in spatio-temporal data, while using cross-modal conditioning to integrate information from multiple domains (visual, textual, frequency).

## Model Architecture

![ViST Architecture](https://example.com/vist_architecture.png)

The ViST architecture consists of several key components:

1. **Multi-view Visual Encoder**: Transforms spatio-temporal data into four complementary visual representations:
   - Spatio-temporal segmentation: Projects data in a time-space grid
   - Gramian Angular Field: Captures temporal correlations using polar coordinates
   - Recurrence Plot: Visualizes recurrence patterns in the time series
   - Spatial Adjacency: Represents node relationships in the spatial domain

2. **Frequency Domain Encoder**: Extracts frequency patterns using FFT, capturing cyclical behaviors

3. **Text Encoder**: Processes descriptive text about the dataset to provide contextual information

4. **Cross-Modal Fusion**: Integrates information from different modalities through attention mechanisms

5. **Dual-path Prediction**:
   - Temporal path: Directly captures sequential dependencies in the data
   - Visual path: Extracts patterns from the visual representations
   - Adaptive gating mechanism: Combines both paths with learned weights

## Key Features

- **Multi-modal Integration**: Combines visual, textual, and frequency domain information
- **Spatial Relationship Modeling**: Explicitly models dependencies between spatial locations
- **Interpretable Visualizations**: The visual encodings provide insights into the data patterns
- **Adaptive Prediction**: Dynamically adjusts the importance of different predictors through gating

## Usage

```python
# Example usage
from arch import ViST
import torch

# Configure the model
config = {
    'seq_len': 12,
    'horizon': 12,
    'num_nodes': 207,
    'c_in': 2,
    'c_out': 1,
    'd_model': 256,
    'image_size': 64,
    # ... other parameters
}

# Initialize model
model = ViST(config)

# Input data: [batch_size, sequence_length, num_nodes, features]
x = torch.randn(32, 12, 207, 2)

# Adjacency matrix (optional)
adj_mx = torch.randn(207, 207)

# Text tokens (optional)
text_tokens = torch.randint(0, 30000, (32, 20))

# Forward pass
predictions = model(x, adj_mx=adj_mx, text_tokens=text_tokens)
# Output shape: [batch_size, horizon, num_nodes, output_features]
```