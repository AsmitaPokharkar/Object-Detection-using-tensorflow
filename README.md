# Object-Detection-using-tensorflow

This project was made as a file submission for my internship with Personifwy. I am very grateful to the mentors and faculty who guided me throughout this . I was able to learn new terminologies and get a hands-on experience with intermediate projects.

It is a complete pipeline for detecting and classifying digits from randomly placed MNIST digits on a 75×75 canvas. The model simultaneously predicts the **digits**(0‑9) and the **bounding box** coordinates of the digit using a single convolutional neural network.

This notebook demonstrates a multi‑task learning approach where the same feature extractor feeds two output heads:  
- **Classification head** (10‑class softmax)  
- **Bounding box regression head** (4‑unit linear output for normalized coordinates)

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
