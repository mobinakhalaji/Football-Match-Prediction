# Football Match Prediction

## Project Overview

This project focuses on analyzing football match data and predicting match outcomes using machine learning techniques.

The main objective is to build a classification model that learns patterns from historical football match data and uses them to predict match results.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest Classifier

## Project Workflow

The project includes the following steps:

1. **Data Loading:** Loading football match data from a CSV file.
2. **Data Preprocessing:** Preparing the dataset for machine learning.
3. **Missing Value Handling:** Using SimpleImputer to handle missing values.
4. **Date Feature Extraction:** Extracting useful information from match dates, including year, month, day, and weekday.
5. **Feature Engineering:** Creating additional features from existing data.
6. **Categorical Encoding:** Converting categorical variables into numerical representations.
7. **Model Training:** Training a Random Forest Classification model.
8. **Model Evaluation:** Evaluating classification performance using appropriate metrics.

## Machine Learning Model

The project uses Random Forest Classifier to predict football match outcomes.

## Evaluation Metrics

The model performance is evaluated using:

- Classification Report
- Precision
- Recall
- F1-Score

## Project Structure

```text
Football-Match-Prediction/
│
├── hw2.py
├── football_matches.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Make sure the dataset file `football_matches.csv` is located in the same directory as `hw2.py`.

Run the following command:

```bash
python hw2.py
```

## Dataset

The project uses a CSV dataset named `football_matches.csv` containing historical football match information.

## Author

Machine Learning / Data Mining Project