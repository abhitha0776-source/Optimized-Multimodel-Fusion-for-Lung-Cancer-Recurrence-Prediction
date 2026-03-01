 Multimodal Fusion for Lung Cancer Recurrence Prediction
* Project Overview

This project focuses on predicting the risk of lung cancer recurrence by combining clinical patient data and CT scan images using machine learning and deep learning techniques. The aim is to assist in early risk assessment after treatment and support better clinical decision-making.

The system uses structured clinical data along with medical image analysis to improve prediction reliability through a multimodal approach.

* Objective

The main objective of this project is to build a predictive model that can estimate the probability of lung cancer recurrence by learning patterns from past patient data and medical images.

* Methodology
1. Clinical Data Processing

Clinical patient data such as tumor stage, invasion status, and treatment details are cleaned and preprocessed.
Categorical features are encoded into numerical format, and important features are selected for training.

An XGBoost classification model is trained on the processed clinical data to predict recurrence risk.

2️. Image-Based Prediction

CT scan images are analyzed using a VGG16-based deep learning model.
The model extracts meaningful visual features from medical images and generates a prediction score.

3️. Multimodal Fusion

Predictions from both the clinical model and the image model are combined using a probability-level fusion approach.
This helps improve overall prediction accuracy compared to using only one data source.

4. Model Evaluation

Each model was evaluated separately before performing multimodal fusion.

* Clinical Model (XGBoost)

Accuracy: ~82%

ROC-AUC: ~0.82

Evaluated using Precision, Recall, F1-Score, and Confusion Matrix

Stratified cross-validation used to maintain balanced class distribution

The clinical model showed stable performance in predicting recurrence risk based on structured patient data.

* Image Model (VGG16)

Accuracy: ~88%

Evaluated using Accuracy, Precision, Recall, and ROC-AUC

Extracted meaningful visual features from CT scan images

The image model achieved higher accuracy, showing strong performance in identifying recurrence-related visual patterns.

* Multimodal Fusion Model

Combined probability outputs from both clinical and image models

Improved overall prediction reliability

Reduced false negatives by leveraging both structured and image data

The results show that integrating clinical data with CT image analysis enhances prediction stability and overall model effectiveness.

5. Deployment

A simple Streamlit web application is developed to demonstrate the model’s predictions.
Users can input patient details, and the system outputs a probability score indicating recurrence risk.

6. Technologies Used

Python

XGBoost

VGG16 (Deep Learning Model)

NumPy, Pandas

Scikit-learn

Matplotlib

Streamlit

7. My Contribution

This project was developed as a group academic project.

My contribution included:

Clinical data preprocessing and feature engineering

Training and evaluation of the XGBoost clinical model

Supporting multimodal fusion implementation

Assisting in Streamlit application integration



8. Conclusion

This project demonstrates how combining clinical data and medical image analysis can improve lung cancer recurrence prediction. The multimodal approach enhances model reliability and shows the potential of AI-based decision support systems in healthcare.
