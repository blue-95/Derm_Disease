# Dermatological Disease Classification

## A Statistical & Machine Learning Pipeline

### Overview

This project implements an end-to-end data analytics and machine learning pipeline to classify six dermatological diseases with overlapping clinical presentations. Using clinical and histopathological features from the UCI Dermatology Dataset, the workflow combines statistical testing, dimensionality reduction, and supervised learning to build a high-accuracy diagnostic model.

The final optimized model achieves 98.18% accuracy, demonstrating the effectiveness of combining feature selection, PCA, and hyperparameter tuning for structured medical data.

⸻

## Problem Statement

Several dermatological conditions—such as Psoriasis and Seborrheic Dermatitis—exhibit similar surface-level symptoms, making diagnosis challenging. This project explores whether underlying histopathological patterns can be leveraged to reliably distinguish between these conditions using data-driven methods.

⸻

## Dataset
	•	Source: UCI Machine Learning Repository – Dermatology Dataset
	•	Features: 34 clinical and histopathological attributes
	•	Target: 6 dermatological disease classes
	•	Challenge: Mixed feature types and missing values in patient age

⸻

## Methodology

1. Data Cleaning & Preprocessing
	•	Converted inconsistent age values (recorded as ?) to numeric format.
	•	Imputed missing age values using the median (35.0) to preserve distribution characteristics.
	•	Validated age distribution using the 1.5×IQR rule, with no significant outliers detected.



2. Statistical Analysis & Feature Engineering
	•	Grouped features into clinical, epidermal histopathological, and dermal histopathological categories.
	•	Performed ANOVA F-tests to evaluate feature discriminative power across disease classes.
	•	Identified key diagnostic markers, including:
	•	band_like_infiltrate
	•	vacuolisation_damage_basal_layer



3. Dimensionality Reduction (PCA)
	•	Applied Principal Component Analysis (PCA) to address multicollinearity among features.
	•	Retained 92% of total variance using 20 principal components.
	•	Visualized class separation using the first two principal components, showing distinct clustering patterns.



4. Machine Learning & Model Optimization

Multiple classifiers were evaluated on both full and PCA-reduced feature sets:
	•	Logistic Regression (PCA): 96.36% accuracy
	•	Support Vector Classifier (PCA): 95.45% accuracy
	•	Random Forest (Default): 97.27% accuracy
	•	Random Forest (Optimized):
	•	Tuned using GridSearchCV (n_estimators, max_depth)
	•	Achieved 98.18% accuracy

⸻

## Key Findings
	•	Histopathological features provide stronger discriminatory power than surface-level clinical attributes.
	•	PCA effectively reduced feature dimensionality while preserving diagnostic signal.
	•	A tuned Random Forest model delivered the most robust performance for this dataset.

⸻

## Tools
	•	Language: Python
	•	Libraries: Pandas, NumPy, Scikit-learn, SciPy, Matplotlib, Seaborn
	•	Environment: Jupyter Notebook
