<div align="center">

# ✌️✋✊ Deep-RPS

**A Real-Time, Deep Learning Rock-Paper-Scissors Computer Vision Game**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-90.32%25-brightgreen.svg)]()

> A custom Convolutional Neural Network (CNN) built entirely from scratch using PyTorch to play Rock, Paper, Scissors in real-time through your webcam.

</div>

---

## ✨ Features

- **Custom CNN Architecture**: A highly optimized 5-block Convolutional Neural Network built specifically for 300x300 image inputs.
- **Mixed Precision Training**: Utilizes `torch.cuda.amp` to maximize GPU utilization and prevent VRAM out-of-memory errors on mid-range GPUs (e.g., RTX 4050).
- **High Accuracy**: Reached **90.32% validation accuracy** in just 10 epochs.
- **Real-Time Inference**: A lightweight OpenCV engine captures live webcam frames, extracts a central Region of Interest (ROI), and runs inference instantly using `torch.no_grad()`.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| 📓 `1_Data_and_Training.ipynb` | The training pipeline. Handles downloading the Laurence Moroney dataset, applying CPU-bound augmentations, defining the CNN architecture, and running the training loop. |
| 🎮 `2_Live_Game.py` | The live game client. A standalone script that loads the trained weights (`deep_rps_model.pth`) and launches the OpenCV webcam interface for real-time predictions. |
| 📦 `requirements.txt` | The list of dependencies required to run the project locally. |

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.11 or newer installed. This project requires a CUDA-enabled NVIDIA GPU for optimal training performance.

Clone the repository and install the dependencies:
```bash
# Create and activate a virtual environment
python -m venv myenv
.\myenv\Scripts\Activate.ps1

# Install requirements (includes PyTorch with CUDA 12.1 support)
pip install -r requirements.txt
```

### 2. Training the Model
Open the Jupyter Notebook and execute all cells sequentially:
- The dataset will be downloaded automatically.
- The training loop will run, displaying live accuracy/loss plots.
- Once complete, the model weights will be saved locally as `deep_rps_model.pth`.

### 3. Playing the Game
Once your model is trained and the `.pth` file is generated, launch the live game:
```bash
python 2_Live_Game.py
```
A webcam window will appear with a green **Region of Interest (ROI)** box. Simply place your hand inside the box to see the model's predictions and confidence scores overlayed dynamically on the screen! 

Press **`Q`** at any time to quit the game.

---

<div align="center">
<i>Built with PyTorch & OpenCV</i>
</div>
