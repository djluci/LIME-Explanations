Student Grade Prediction & Model Interpretability with LIME
A machine learning project that trains and interprets classification models to predict whether a student will score above 90, using Logistic Regression, Neural Networks, and LIME explanations.

Overview
This project explores both model training and model interpretability on a real-world student performance dataset. The core goals are:

Train and evaluate a Logistic Regression model and a Neural Network (MLP) classifier
Perform hyperparameter tuning for the neural network
Use LIME (Local Interpretable Model-agnostic Explanations) to explain individual predictions
Compare how LIME explanations differ between a linear model and a non-linear model


Repository Structure
.
├── Assignment_1_LIME.ipynb   # Main notebook with all code and analysis
├── grades_dataset.csv        # Student performance dataset (~10,000 samples)
└── README.md

Dataset
grades_dataset.csv — ~10,000 student records with 29 features and 1 binary target label.
Feature CategoryExamplesAcademicReadingScore, WritingScoreDemographicGender_female, EthnicGroup_group A–EFamily BackgroundNrSiblings, ParentEduc_*, ParentMaritalStatus_*, IsFirstChild_noLifestylePracticeSport_*, WklyStudyHours_*, TransportMeans_privateSchoolLunchType_free/reduced, TestPrep_completedTargetGrade>90 (1 = scored above 90, 0 = did not)
All categorical features are already one-hot encoded.

Workflow
1. Data Loading & Preprocessing

Load CSV with pandas; separate features (X) and labels (y)
Split into train (60%) / validation (20%) / test (20%) using train_test_split
Standardize features with StandardScaler (fit on train only)

2. Logistic Regression

Train a baseline LogisticRegression model (scikit-learn defaults)
Evaluate with Accuracy and AUC on train and test sets
Visualize model weights as a bar chart to understand feature importance globally

3. Neural Network (MLP Classifier)

Train a MLPClassifier with max_iter=20000
Hyperparameter search over hidden layer configurations:

(10,) · (20,) · (10, 20) · (10, 20, 20)


Extended search varying activation functions (relu, tanh, logistic) and deeper architectures
Best configuration selected by highest validation AUC

4. LIME Explanations

Use LimeTabularExplainer to generate local explanations for 10 randomly selected test instances
Settings: num_features=5, num_samples=50,000
Explanations generated for both the Logistic Regression model and the best Neural Network
Comparison of how feature importance varies between a linear vs. non-linear model


Key Findings

Logistic Regression produces consistent, globally interpretable weights; LIME explanations closely mirror the weight plot, confirming LIME is working correctly.
Neural Networks capture non-linear relationships, so LIME explanations vary more across instances — reflecting the model's complexity.
Hyperparameter tuning improved validation AUC over the default single-hidden-layer baseline.
The most influential features for grade prediction relate to academic scores (ReadingScore, WritingScore) and study habits (WklyStudyHours, TestPrep_completed).


Requirements
pandas
numpy
scikit-learn
lime
matplotlib
Install all dependencies:
bashpip install pandas numpy scikit-learn lime matplotlib

Usage

Clone the repository and ensure grades_dataset.csv is in the same directory as the notebook.
Install dependencies (see above).
Open and run Assignment_1_LIME.ipynb top-to-bottom in Jupyter.

bashjupyter notebook Assignment_1_LIME.ipynb

References

LIME Documentation
scikit-learn MLPClassifier
Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD 2016.
