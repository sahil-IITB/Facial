# Facial Expression Recognition on FER-2013 Using CNN in PyTorch

## 📌 Overview
This project implements a **Convolutional Neural Network (CNN)** for **Facial Expression Recognition (FER)** on the **FER-2013 dataset**.  
The model classifies grayscale facial images into **seven emotions**:

- Angry 😠  
- Disgust 😖  
- Fear 😨  
- Happy 😀  
- Sad 😢  
- Surprise 😮  
- Neutral 😐  

The implementation includes:
- **Custom dataset loader**
- **Data augmentation**
- **CNN architecture with BatchNorm & Dropout**
- **Training with Adam optimizer & learning rate scheduling**
- **Performance evaluation with accuracy/loss curves & confusion matrix**

---

## 📂 Dataset
- **Source:** [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)  
- **Format:** CSV with each image as a `48×48` grayscale pixel array (space-separated values)  
- **Splits:**
  - Training: **28,709** samples
  - Validation (PublicTest): **3,589** samples
  - Test (PrivateTest): **3,589** samples

**Emotion Labels**
| Label | Emotion   |
|-------|-----------|
| 0     | Angry     |
| 1     | Disgust   |
| 2     | Fear      |
| 3     | Happy     |
| 4     | Sad       |
| 5     | Surprise  |
| 6     | Neutral   |

---

## ⚙️ Data Preprocessing
### **Augmentations**
- **Training:**
  - Random horizontal flip
  - Random rotation (±10°)
  - Normalization to range `[-1, 1]`
- **Validation/Test:**
  - Normalization to range `[-1, 1]`

### **Dataloader**
- Batch size: **64**
- `DeviceDataLoader` wrapper for **GPU support**

---

## 🏗 Model Architecture
The model consists of **4 convolutional blocks** followed by **2 fully connected layers**.

| Layer                        | Output Shape   | Details |
|------------------------------|---------------|---------|
| Input                        | [1, 48, 48]   | Grayscale image |
| Conv2D + BN + ReLU           | [32, 48, 48]  | Kernel: 3×3, padding=1 |
| MaxPool2D                    | [32, 24, 24]  | Pool: 2×2 |
| Conv2D + BN + ReLU           | [64, 24, 24]  |          |
| MaxPool2D                    | [64, 12, 12]  |          |
| Conv2D + BN + ReLU           | [128, 12, 12] |          |
| MaxPool2D                    | [128, 6, 6]   |          |
| Conv2D + BN + ReLU           | [256, 6, 6]   |          |
| MaxPool2D                    | [256, 3, 3]   |          |
| Flatten                      | 2304          |          |
| FC1 + ReLU + Dropout (0.4)   | 256           |          |
| FC2                          | 7             | Logits   |

---

## 🏋️ Training Setup
- **Loss Function:** `CrossEntropyLoss`
- **Optimizer:** Adam (`lr = 0.001`)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=2)
- **Epochs:** 50
- **Model Checkpoint:** Saves best model based on validation accuracy

---

## 📊 Results
### **Accuracy & Loss Curves**
![Training Metrics](plots/train_metrics_lr0.001_ep50.png)

### **Confusion Matrix**
![Confusion Matrix](plots/conf_matrix_lr0_001_ep50.png)

### **Test Set Performance**
- **Accuracy:** `63.33%`
- **Loss:** `1.1434`

---

## 📁 Repository Structure
