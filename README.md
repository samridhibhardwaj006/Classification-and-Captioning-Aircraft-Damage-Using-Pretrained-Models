# Classification-and-Captioning-Aircraft-Damage-Using-Pretrained-Models

📖 About

This is the Final Project of the IBM AI Engineering Professional Certificate on Coursera. It tackles a real-world aviation safety problem: automatically detecting and describing aircraft damage using Deep Learning and Transformer-based models.

The project is split into two parts — a classification task (dent vs. crack detection) and an image captioning task (generating natural language descriptions of the damage).


⏱️ Estimated time: 90 minutes




🎯 Project Objectives


✅ Classify aircraft damage images into "dent" or "crack" using VGG16 (Transfer Learning)
✅ Preprocess and augment image data using Keras ImageDataGenerator
✅ Evaluate model performance with accuracy curves & visualizations
✅ Generate image captions and summaries using the BLIP Transformer model
✅ Implement a custom Keras layer within the pipeline



🧠 Project Architecture

Aircraft Image
      │
      ▼
┌─────────────────────────┐     ┌───────────────────────────┐
│   PART 1: Classification│     │  PART 2: Image Captioning  │
│                         │     │                            │
│  VGG16 (pretrained)     │     │  BLIP Transformer Model    │
│  + Custom Dense Layers  │     │  (HuggingFace)             │
│         ↓               │     │         ↓                  │
│  "dent" or "crack" 🔍   │     │  Natural language caption  │
└─────────────────────────┘     └───────────────────────────┘


📋 Task Breakdown

TaskDescriptionTask 1Create valid_generator for validation dataTask 2Create test_generator for test dataTask 3Load the VGG16 pretrained modelTask 4Compile the model with Adam optimizerTask 5Train the model (5 epochs)Task 6Plot accuracy curves (train vs. validation)Task 7Visualize predictions on test imagesTask 8Implement a custom Keras layer helper functionTask 9Generate an image caption using BLIPTask 10Generate an image summary using BLIP


📂 Dataset


Source: Roboflow Aircraft Damage Detection Dataset
License: CC BY 4.0
Structure:


aircraft_damage_dataset_v1/
├── train/
│   ├── dent/
│   └── crack/
├── valid/
│   ├── dent/
│   └── crack/
└── test/
    ├── dent/
    └── crack/


🛠️ Tech Stack

ToolPurposePython 3.xCore languageTensorFlow / KerasModel building & trainingVGG16Pretrained CNN for feature extractionBLIP (HuggingFace)Image captioning & summarizationPyTorchBackend for Transformer modelMatplotlibVisualizing accuracy & predictionsImageDataGeneratorReal-time image preprocessing & augmentation


📊 Model Configuration

pythonbatch_size = 32
n_epochs   = 5
img_size   = (224, 224)   # VGG16 input size
optimizer  = Adam
loss       = binary_crossentropy


📂 File Structure

📦 IBM-AI-Engineering-Final-Project
 ┣ 📓 Final_Project_Classification_and_Captioning.ipynb
 ┗ 📄 README.md


📜 Course


🎓 Final Project of the IBM AI Engineering Professional Certificate
offered by IBM on Coursera.




🙋 Author

Samridhi Bhardwaj


