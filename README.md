# ✈️ Aircraft Damage Classification and Image Captioning

A deep learning project that combines computer vision and natural language processing to analyze aircraft damage images. The project uses transfer learning with VGG16 to classify aircraft damage and a BLIP transformer model to generate image captions and summaries.

---

## 📖 Project Overview

Aircraft inspections are essential for ensuring aviation safety, but manual inspection processes can be time-consuming and prone to human error. This project explores how artificial intelligence can assist aircraft damage assessment through automated image analysis.

The project consists of two major components:

### 1. Aircraft Damage Classification

A VGG16-based transfer learning model is trained to classify aircraft damage into:

* Dent
* Crack

### 2. Image Captioning and Summarization

A BLIP (Bootstrapping Language-Image Pretraining) transformer model is used to generate:

* Natural language image captions
* Descriptive image summaries

This creates a complete pipeline capable of both recognizing aircraft damage and describing it in human-readable language.

---

## 🎯 Objectives

* Classify aircraft damage using deep learning.
* Apply transfer learning using a pretrained VGG16 network.
* Evaluate model performance using accuracy and loss metrics.
* Generate image captions using BLIP.
* Generate descriptive summaries from aircraft images.
* Demonstrate the integration of computer vision and transformer-based NLP models.

---

## 🧠 Aircraft Damage Classification

### Model Architecture

The classification model uses:

* VGG16 (ImageNet Pretrained)
* Transfer Learning
* Feature Extraction
* Dense Classification Layers
* Binary Classification Output

### Workflow

1. Download and preprocess aircraft image dataset.
2. Normalize image data using ImageDataGenerator.
3. Extract image features using VGG16.
4. Train a custom classifier head.
5. Evaluate classification performance.

---

## 📊 Classification Results

### Training and Validation Loss

![Training Loss](images/training_loss.png)

### Training and Validation Accuracy

![Training Accuracy](images/training_accuracy.png)

### Prediction Example

![Prediction Example](images/prediction_example.png)

---

## 🤖 Image Captioning and Summarization

### Model

The project uses:

* BLIP (Bootstrapping Language-Image Pretraining)
* Hugging Face Transformers
* Image-to-Text Generation

### Workflow

1. Load aircraft image.
2. Process image with BLIP processor.
3. Generate caption.
4. Generate detailed summary.

---

## 📝 Caption Generation Results

### Generated Caption

![Generated Caption](images/generated_caption.png)

### Generated Summary

![Generated Summary](images/generated_summary.png)

---

## 🛠️ Technologies Used

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Programming Language    |
| TensorFlow   | Deep Learning Framework |
| Keras        | Model Development       |
| VGG16        | Transfer Learning       |
| Transformers | NLP Framework           |
| BLIP         | Image Captioning        |
| NumPy        | Numerical Computing     |
| Matplotlib   | Visualization           |
| Pillow       | Image Processing        |

---

## 📁 Project Structure

```text
aircraft-damage-classification-captioning/
│
├── README.md
├── requirements.txt
├── main.py
│
├── src/
│   ├── data_loader.py
│   ├── classifier.py
│   ├── visualization.py
│   └── captioning.py
│
├── images/
│
└── models/
```

---

## 🚀 Future Improvements

* Experiment with ResNet50 and EfficientNet architectures.
* Expand the dataset with additional damage categories.
* Deploy the model as a web application.
* Integrate object detection for localized damage analysis.
* Fine-tune larger vision-language models for richer image descriptions.

---

## 👨‍💻 Author

**Samridhi Bhardwaj**

Originally developed as part of an AI and Deep Learning project and later refactored into a structured portfolio project.
