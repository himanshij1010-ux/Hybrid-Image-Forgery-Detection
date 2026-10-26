# Hybrid Image Forgery Detection Using CNN and Random Forest

## 1. Project Overview

This project implements an experimental image classification system using a hybrid Deep Learning and Machine Learning approach. A Convolutional Neural Network (CNN) extracts visual features from an image, and a Random Forest classifier uses these features to classify the image as Real or Fake.

A Streamlit web application allows users to upload images and view the model's prediction and class scores.

The system is designed for educational and experimental purposes. Its predictions are not definitive proof that an image is authentic or manipulated.

## 2. Objectives

- Develop an image classification pipeline for real and potentially manipulated images.
- Extract visual features using a trained CNN.
- Combine CNN-based feature extraction with a Random Forest classifier.
- Evaluate model performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Build a user-friendly web application for image analysis.
- Demonstrate how deep learning and traditional machine learning can work together.

## 3. Technologies Used

- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow and Keras
- **Numerical Processing:** NumPy
- **Machine Learning:** Scikit-learn
- **Classification Algorithm:** Random Forest
- **Image Processing:** Pillow
- **Model Serialization:** Joblib
- **Web Application:** Streamlit

## 4. System Workflow

The project follows these steps:

1. Load the processed dataset containing real and fake images.
2. Resize images to 128 × 128 pixels and prepare them for model processing.
3. Assign class labels: Fake = 0 and Real = 1.
4. Split the dataset into training and validation subsets using a stratified split.
5. Use the trained CNN to extract a 64-dimensional feature representation from each image.
6. Train the Random Forest classifier using the extracted features and corresponding labels.
7. Evaluate the hybrid model using the validation dataset.
8. Save the trained CNN and Random Forest models.
9. Launch the Streamlit application.
10. Allow the user to upload an image and view the predicted class and class scores.

## 5. Dataset Description

The processed dataset contains 12,614 images.

| Image Category | Number of Images |
|---|---:|
| Real images | 7,491 |
| Fake images | 5,123 |
| **Total** | **12,614** |

The dataset is organized into real and fake image categories. The training code assigns the label 0 to fake images and 1 to real images.

The dataset is divided into training and validation subsets using a stratified 80:20 split. Stratification helps preserve the class proportions in both subsets.

The validation results reported in this project are preliminary and should not be treated as an independent final test of real-world performance.

## 6. Model Architecture

### 6.1 Convolutional Neural Network (CNN)

A Convolutional Neural Network is a deep learning architecture commonly used for image processing and classification.

In this project, the CNN processes an image and learns useful visual patterns. A 64-unit dense layer is used to generate a 64-dimensional feature vector for the hybrid classifier.

The CNN feature extractor converts each processed image into numerical values that represent learned visual information.

### 6.2 Random Forest Classifier

Random Forest is a supervised machine learning algorithm that combines multiple decision trees to make a prediction.

In this project, the Random Forest classifier receives the 64-dimensional feature vectors extracted by the CNN. It uses the learned patterns to classify an image into one of two categories:

- Fake
- Real

The classifier is trained with 200 decision trees, a maximum tree depth of 20, a minimum of 2 samples per leaf, and balanced class weights.

### 6.3 Hybrid Approach

The hybrid model combines the strengths of two approaches:

- **CNN:** Extracts visual features from images.
- **Random Forest:** Uses the extracted features to perform the final classification.

This design separates feature extraction from the final classification stage.

## 7. Validation Results

The hybrid CNN and Random Forest model achieved **70.79% validation accuracy** on the current validation split.

### Classification Metrics

| Metric | Fake | Real |
|---|---:|---:|
| Precision | 0.62 | 0.78 |
| Recall | 0.71 | 0.71 |
| F1-score | 0.66 | 0.74 |

### Interpretation

- **Accuracy:** The overall proportion of correctly classified validation images was approximately 70.79%.
- **Precision:** Measures how many images predicted as a particular class actually belong to that class.
- **Recall:** Measures how many images belonging to a class were correctly identified.
- **F1-score:** Combines precision and recall into a single metric.

These are preliminary validation results from the current data split. They do not establish the model's performance on all image types or on an independent dataset.

## 8. How to Run the Project

### Prerequisites

Install Python 3.11 and ensure that the project files are available on your computer.

### Step 1: Open the Project Folder

Open the project folder in Visual Studio Code and activate the existing virtual environment if it is not already active.

On Windows PowerShell, run:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

Run the following command:

```powershell
python -m pip install -r requirements.txt
```

### Step 3: Train the Hybrid Classifier

Ensure that the pretrained CNN model `forgery_detection_model_v3.keras` and the processed dataset are available.

Run:

```powershell
python train_model.py
```

This script extracts CNN features, trains the Random Forest classifier, evaluates the hybrid model, and saves the classifier.

### Step 4: Evaluate the CNN Model

Run:

```powershell
python evaluate_model.py
```

This script evaluates the CNN model using the validation data.

### Step 5: Run the Web Application

Run:

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal. It is commonly:

http://localhost:8501

Upload an image to view the predicted class and the classifier's estimated Fake and Real scores.

### Step 6: Run Prediction Diagnostics

Run:

```powershell
python check_predictions.py
```

This script checks predictions on a sample of real images from the validation split.

**Note:** Training is not required every time the application is launched. The saved CNN and Random Forest models can be loaded by the application when the required files are present.

## 9. Important Project Files

| File or Folder | Purpose |
|---|---|
| `app.py` | Streamlit web application for image classification |
| `train_model.py` | CNN feature extraction and Random Forest training |
| `evaluate_model.py` | CNN validation evaluation |
| `check_predictions.py` | Diagnostic checks on real validation images |
| `prepare_dataset.py` | Dataset preparation script |
| `forgery_detection_model_v3.keras` | Trained CNN model used for feature extraction |
| `hybrid_forgery_classifier.pkl` | Saved Random Forest classifier |
| `requirements.txt` | Python package dependencies |
| `dataset/` | Dataset and related image folders |
| `README.md` | Project documentation |

## 10. Limitations

- The model may incorrectly classify genuine photographs as fake or manipulated images as real.
- The dataset may not represent all types of real-world images and manipulations.
- Performance may vary with image quality, compression, editing techniques, and image sources.
- Random Forest class scores are estimates produced by the classifier and are not necessarily calibrated probabilities of authenticity.
- A predicted Fake label does not prove that an image was AI-generated or manipulated.
- A predicted Real label does not guarantee that an image is genuine.
- The current validation results do not replace testing on an independent dataset.
- The system should not be used as the sole basis for forensic, legal, or other high-stakes decisions.

## 11. Future Enhancements

- Evaluate the system on an independent dataset containing unseen images.
- Improve performance on different image sources and manipulation techniques.
- Compare different CNN architectures and machine learning classifiers.
- Add image tampering localization to identify potentially manipulated regions.
- Incorporate explainability methods to help users understand predictions.
- Improve uncertainty reporting and probability calibration.
- Expand the web interface with evaluation visualizations and more detailed reports.

## 12. Conclusion

This project demonstrates an experimental hybrid image classification system that combines CNN-based feature extraction with a Random Forest classifier.

The system processes images, extracts a 64-dimensional feature representation, and predicts whether an image belongs to the Real or Fake class. A Streamlit application provides an interface for uploading images and viewing predictions.

The hybrid model achieved 70.79% accuracy on the current validation split. While this demonstrates a working implementation, further testing, independent evaluation, and model improvements are needed before the system can be considered reliable for real-world image-forgery detection.

The project provides a practical demonstration of combining deep learning and traditional machine learning for image classification.