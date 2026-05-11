
Student Grade Prediction & Model Interpretability with LIME

A machine learning project that trains and interprets classification models to predict whether a student will score above 90, using Logistic Regression, Neural Networks, and LIME explanations.

Overview
This project explores both model training and model interpretability on a real-world student performance dataset. The core goals are:

Train and evaluate a Logistic Regression model and a Neural Network (MLP) classifier
Perform hyperparameter tuning for the neural network
Use LIME (Local Interpretable Model-agnostic Explanations) to explain individual predictions
Compare how LIME explanations differ between a linear and a non-linear model


Repository Structure
.
├── Assignment_1_LIME.ipynb   # Main notebook with all code and analysis
├── grades_dataset.csv        # Student performance dataset (~10,000 samples)
└── README.md

Dataset
grades_dataset.csv — ~10,000 student records with 29 features and 1 binary target label. All categorical features are already one-hot encoded.
Feature CategoryExamplesAcademicReadingScore, WritingScoreDemographicGender_female, EthnicGroup_group A–EFamily BackgroundNrSiblings, ParentEduc_*, ParentMaritalStatus_*, IsFirstChild_noLifestylePracticeSport_*, WklyStudyHours_*, TransportMeans_privateSchoolLunchType_free/reduced, TestPrep_completedTargetGrade>90 (1 = scored above 90, 0 = did not)

Workflow
1. Data Loading & Preprocessing
Split into train (60%) / validation (20%) / test (20%) using train_test_split. Features standardized with StandardScaler fit on the train set only.
2. Logistic Regression
Baseline model evaluated with Accuracy and AUC. Model weights visualized as a bar chart for global feature importance.
3. Neural Network (MLP Classifier)
Hyperparameter search over hidden layer configurations — (10,), (20,), (10, 20), (10, 20, 20) — plus an extended search varying activation functions (relu, tanh, logistic). Best model selected by validation AUC.
4. LIME Explanations
LimeTabularExplainer used to explain 10 randomly selected test instances (num_features=5, num_samples=50,000) for both models, with a comparison of how local explanations differ between linear and non-linear architectures.

Requirements
bashpip install pandas numpy scikit-learn lime matplotlib

Usage
Ensure grades_dataset.csv is in the same directory as the notebook, then:
bashjupyter notebook Assignment_1_LIME.ipynb

References

LIME Documentation
scikit-learn MLPClassifier
Ribeiro et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD 2016.
