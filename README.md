# Object-Detection-using-tensorflow
  MNIST Digit Detection

A complete pipeline for detecting and classifying digits from randomly placed MNIST digits on a 75×75 canvas. The model simultaneously predicts the **digits**(0‑9) and the **bounding box** coordinates of the digit using a single convolutional neural network.

This notebook demonstrates a multi‑task learning approach where the same feature extractor feeds two output heads:  
- **Classification head** (10‑class softmax)  
- **Bounding box regression head** (4‑unit linear output for normalized coordinates)

---

##  Overview :
The notebook:
1. **Loads MNIST** via `tensorflow_datasets`.
2. **Creates synthetic data** by placing each 28×28 digit at a random position inside a 75×75 canvas, producing both the image and the ground‑truth bounding box.
3. **Defines a CNN** with three convolutional + pooling blocks followed by a dense layer.
4. **Splits the dense layer** into two output branches.
5. **Compiles the model** with two losses: `categorical_crossentropy` for classification and `mse` for bounding box regression.
6. **Trains for 5 epochs** using a custom batch‑shuffling pipeline.
7. **Evaluates** and visualizes predictions (class + bounding box) alongside ground truth.

---

## 🧠 Model Architecture :
The model is built with Keras Functional API:

| Layer                     | Output Shape     | Params   |
|---------------------------|------------------|----------|
| Input (75×75×1)           | (None, 75, 75, 1)| 0        |
| Conv2D (16, 3×3, ReLU)    | (None, 73, 73, 16)| 160     |
| AveragePooling2D (2×2)    | (None, 36, 36, 16)| 0       |
| Conv2D (32, 3×3, ReLU)    | (None, 34, 34, 32)| 4 640   |
| AveragePooling2D (2×2)    | (None, 17, 17, 32)| 0       |
| Conv2D (64, 3×3, ReLU)    | (None, 15, 15, 64)| 18 496  |
| AveragePooling2D (2×2)    | (None, 7, 7, 64) | 0       |
| Flatten                   | (None, 3136)     | 0       |
| Dense (128, ReLU)         | (None, 128)      | 401 536 |
| **Classification Dense**  | (None, 10)       | 1 290   |
| **Bounding Box Dense**    | (None, 4)        | 516     |

**Total params**: 426 638 (≈1.63 MB) – all trainable.

---

##  Training Setup :

- **Optimizer**: Adam (default learning rate)
- **Loss**:
  - Classification: `categorical_crossentropy`
  - Bounding box: `mean squared error` (MSE)
- **Metrics**:
  - Classification: `accuracy`
  - Bounding box: `mse`
- **Batch size**: 64 × number of replicas (distributed strategy used)
- **Epochs**: 5
- **Data augmentation**: random placement of each digit (position varies per epoch)
- **Preprocessing**: images scaled to `[0,1]`, bounding boxes normalized to `[0,1]` relative to canvas size.

### Data Generation
Each training sample is produced on the fly:
- A random `(xmin, ymin)` offset (0–48) is chosen.
- The 28×28 MNIST digit is placed at that offset inside a 75×75 canvas.
- Bounding box coordinates are converted to normalized values:  
  `xmin/75, ymin/75, (xmin+28)/75, (ymin+28)/75`.

The dataset is shuffled, batched, and prefetched for performance.

---

## 📈 Results (after 5 epochs)

| Metric                     | Training (last epoch) | Validation |
|----------------------------|----------------------|------------|
| Classification Loss        | ≈0.79                | 0.73       |
| Classification Accuracy    | ≈75.6%               | 75.4%      |
| Bounding Box MSE           | ≈0.0185              | 0.0202     |

The model learns to localize digits reasonably well and classify most of them correctly, despite the small training budget (5 epochs). With more epochs, accuracy and IoU would improve.

---

## How to Run :

### 1. Install dependencies

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib Pillow
```

### 2. Open the notebook

```bash
jupyter notebook "Object Detection using tensorflow.ipynb"
```

Or run in Google Colab – the notebook is designed to work out‑of‑the‑box there.

### 3. Execute cells sequentially

The notebook will download MNIST automatically via `tensorflow_datasets`.

---

##  Extensions & Improvements :

- **Increase epochs** – The model is still improving; 20–50 epochs would raise accuracy >90%.
- **Add IoU metric** – Currently computed manually; can be added as a custom Keras metric.
- **Data augmentation** – Random flips, small rotations could improve generalization.
- **Use a deeper backbone** – Replace the custom CNN with MobileNetV2 or ResNet50 (pretrained).
- **Non‑maximum suppression** – Not needed here (single object per image), but useful for multi‑object detection.
---
