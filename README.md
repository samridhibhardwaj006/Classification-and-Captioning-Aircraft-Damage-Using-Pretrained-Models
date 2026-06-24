# ✈️ Aircraft Damage Classification and Image Captioning

A deep learning project that combines computer vision and natural language processing to automate aircraft damage analysis. The system uses transfer learning with VGG16 to classify aircraft damage and a BLIP transformer model to generate natural language captions and summaries from aircraft images.

---

## 📖 Project Overview

Aircraft inspections play a critical role in aviation safety and maintenance. Traditional inspection methods rely heavily on manual visual assessment, which can be time-consuming and susceptible to human error.

This project demonstrates how artificial intelligence can assist aircraft maintenance workflows through:

* Automated aircraft damage classification
* AI-generated image descriptions
* Vision-language understanding

The solution integrates a convolutional neural network for image classification and a transformer-based model for image captioning and summarization.

---

## 🎯 Objectives

* Classify aircraft damage into **Dent** and **Crack** categories.
* Apply transfer learning using a pretrained VGG16 architecture.
* Evaluate model performance using accuracy and loss metrics.
* Generate natural language image captions using BLIP.
* Generate descriptive image summaries from aircraft inspection images.
* Explore the integration of Computer Vision and Natural Language Processing within a single workflow.

---

## 🧠 Aircraft Damage Classification

### Model Architecture

The classification pipeline is built using:

* VGG16 (ImageNet Pretrained)
* Transfer Learning
* Feature Extraction
* Fully Connected Classification Layers
* Binary Classification Output

### Classification Workflow

1. Download and preprocess aircraft damage images.
2. Normalize image data using ImageDataGenerator.
3. Extract visual features using VGG16.
4. Train a custom classification head.
5. Evaluate performance on unseen test images.

### Damage Categories

* Dent
* Crack

---

## 📊 Model Performance

### Training Loss

![Training Loss](images/training_loss.png)

The training loss curve illustrates how the model's error decreases throughout the learning process.

### Validation Loss

![Validation Loss](images/validation_loss.png)

Validation loss provides insight into the model's ability to generalize to unseen aircraft images.

### Accuracy Curve

![Accuracy Curve](images/training_accuracy.png)

Training and validation accuracy curves were monitored throughout training to evaluate model learning and convergence.

---

## 🔍 Prediction Example

### Aircraft Damage Prediction

![Prediction Example](images/prediction_example.png)

The model successfully predicts the damage category of aircraft inspection images and compares predictions against ground-truth labels.

---

## 🤖 Image Captioning and Summarization

### BLIP Transformer Model

The project utilizes the BLIP (Bootstrapping Language-Image Pretraining) model from Hugging Face Transformers.

BLIP combines visual understanding and language generation to produce human-readable descriptions directly from images.

### Captioning Workflow

1. Load aircraft image.
2. Process image using BLIP Processor.
3. Generate descriptive caption.
4. Generate detailed summary.
5. Return human-readable output.

---

## 📝 Generated Caption

![Caption Output](images/generated_caption.png)

The BLIP model generates natural language captions that describe the visual content of aircraft inspection images.

---

## 📄 Generated Summary

![Summary Output](images/generated_summary.png)

The summarization output provides a more detailed interpretation of image content and observed damage characteristics.

---

## 🛠️ Technologies Used

| Technology                | Purpose                          |
| ------------------------- | -------------------------------- |
| Python                    | Programming Language             |
| TensorFlow                | Deep Learning Framework          |
| Keras                     | Model Development                |
| VGG16                     | Transfer Learning                |
| Hugging Face Transformers | NLP Framework                    |
| BLIP                      | Image Captioning & Summarization |
| NumPy                     | Numerical Computing              |
| Matplotlib                | Visualization                    |
| Pillow                    | Image Processing                 |

---

## 📁 Repository Structure

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
│   ├── training_loss.png
│   ├── validation_loss.png
│   ├── accuracy_curve.png
│   ├── prediction_example.png
│   ├── caption_output.png
│   └── summary_output.png
│
└── models/
```

---

## 💡 Key Learnings

Through this project, I gained practical experience with:

* Transfer Learning
* Computer Vision
* Deep Learning Model Training
* Image Classification
* Hugging Face Transformers
* Vision-Language Models
* Image Caption Generation
* Model Evaluation and Visualization
* Building End-to-End AI Pipelines

---

## 🚀 Future Improvements

* Experiment with ResNet50 and EfficientNet architectures.
* Expand the dataset with additional aircraft damage categories.
* Implement object detection for localized damage identification.
* Deploy the solution as a web application.
* Fine-tune larger vision-language models for richer descriptions.
* Integrate multimodal large language models for maintenance reporting.

---

## 👨‍💻 Author

**Samridhi Bhardwaj**

Originally developed as part of an AI and Deep Learning project and later refactored into a structured portfolio project focused on computer vision and vision-language AI systems.
