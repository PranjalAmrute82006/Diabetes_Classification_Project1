# Diabetes Classification using Machine Learning

## Project Overview

- This project focuses on predicting whether a person is likely to have diabetes using Machine Learning classification techniques.
- The project uses the Pima Indians Diabetes Dataset and applies data preprocessing, exploratory data analysis, model training, and      evaluation to build a diabetes classification system.

## Objective
The main objective of this project is to develop a Machine Learning model that can classify patients into two categories:

- `0` - No Diabetes
- `1` - Diabetes

The project aims to:

- Analyze the diabetes dataset
- Perform data preprocessing
- Explore important features
- Train classification models
- Evaluate model performance
- Compare different classification algorithms
- Predict diabetes outcomes for new patient data

## Dataset

The project uses the **Pima Indians Diabetes Dataset**.

Records : 768
Columns : 9

## Algorithms

1. Logistic Regression

Logistic Regression is a supervised Machine Learning classification algorithm used to predict the probability of a binary outcome.

In this project, it is used to classify patients as diabetic or non-diabetic.

2. Random Forest Classifier

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision trees to improve prediction performance and reduce overfitting.

It is used to classify diabetes outcomes based on the available patient features.

## Technologies

The project was developed using:

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Evaluation Metrics

The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Cross Validation


## Results

| Model | Accuracy | Precision | Recall | F1 Score |

| Logistic Regression | 77.27% | 69.23% | 65.45% | 67.29% |
| Random Forest | 88.31% | 81.36% | 87.27% | 84.21% |

Random Forest achieved the best overall performance with an accuracy of **88.31%**.

## Prediction

The trained model predicts the diabetes outcome for a patient based on input features :

- `0` → No Diabetes
- `1` → Diabetes


## How to Run

 1. Clone the Repository

Clone this repository to your local system.

2. open the project Folder
cd Diabetes_Classification_Project
 
3. Install the required python libraries using:
 pip install -r requirements.txt

4. open the Jupyter Notebook
Diabetes_Classification.ipynb

5. Run the Notebook

## Project Structure 

Diabetes_Classification_Project/
│
├── Diabetes_Classification.ipynb
├── diabetes.csv
├── Project_Report.docx
├── README.md
├── requirements.txt
└── screenshot.png