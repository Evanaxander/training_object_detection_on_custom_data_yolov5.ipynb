Here is a professional README for your GitHub repository based on the notebook content.

# YOLOv5 Custom Object Detection: Helmet Detection

This repository contains a comprehensive pipeline for training a custom YOLOv5 object detection model. The provided notebook demonstrates the end-to-end process from environment setup to training a detector on a custom "Helmet Detection" dataset.

## Project Overview

The project utilizes the **Ultralytics YOLOv5** architecture to identify safety helmets in images. It features seamless integration with **Roboflow** for dataset management and versioning.

### Key Features

* **Custom Data Training**: Step-by-step guide to downloading datasets and configuring them for YOLOv5.
* **Model Configuration**: Automatic generation of a custom `yolov5s.yaml` architecture modified for specific class counts.
* **Transfer Learning**: Uses pre-trained weights (`yolov5s.pt`) to accelerate training and improve accuracy.

## Getting Started

### Prerequisites

* Python 3.x
* PyTorch (with CUDA support for GPU acceleration)
* Roboflow API Key

### Installation

1. Clone the YOLOv5 repository:
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5

```


2. Install dependencies:
```bash
pip install -qr requirements.txt
pip install roboflow

```



## Usage

### 1. Download Dataset

The project uses a helmet detection dataset hosted on Roboflow. You can load it using the following Python snippet:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("evanaxander").project("helmet-detection-jsg4w")
dataset = project.version(1).download("yolov5")

```

### 2. Configure Model

The model configuration is automatically adjusted based on the dataset's classes (e.g., `nc: 1` for helmet detection).

### 3. Training

Start training the model with specific image size, batch size, and epoch count:

```bash
python train.py --img 416 --batch 16 --epochs 50 --data {dataset.location}/data.yaml --cfg ./models/custom_yolov5s.yaml --weights yolov5s.pt --cache

```

## Training Arguments

* `--img`: Input image size (default: 416).
* `--batch`: Training batch size (default: 16).
* `--epochs`: Number of training epochs.
* `--data`: Path to the dataset `data.yaml` file.
* `--cfg`: Path to the model configuration file.
* `--weights`: Initial weights (supports local paths or Ultralytics Google Drive weights).

## Acknowledgments

* [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the object detection framework.
* [Roboflow](https://roboflow.com) for dataset hosting and simplified API access.
