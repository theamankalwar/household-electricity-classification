
# Household Electricity Consumption Analysis and Classification

A Python-based data science project that analyzes household electricity consumption and classifies observations into **High** and **Low** consumption categories using machine learning.

## Overview

This project uses the UCI Individual Household Electric Power Consumption dataset to explore electricity usage patterns, perform data preprocessing, visualize measurements, and compare classification models.

An interactive Streamlit application allows users to enter electricity measurements and view a predicted consumption class.

## Features

- Data inspection and preprocessing using Pandas and NumPy
- Handling missing values and checking duplicate records
- Electricity consumption visualizations using Matplotlib
- High/Low consumption classification
- Decision Tree and Logistic Regression models
- Model evaluation using accuracy, confusion matrix, precision, recall, and F1-score
- Interactive Streamlit dashboard
- Basic rule-based chatbot in the Jupyter notebook

## Dataset

**Source:** [UCI Machine Learning Repository — Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption)

The original dataset contains measurements collected at approximately one-minute intervals from a single household.

Download the dataset from UCI and place `household_power_consumption.txt` inside the local `data/` directory.

The raw dataset is not included in this repository.

## Methodology

1. Load and inspect the dataset using Pandas.
2. Examine missing values and duplicate records.
3. Handle missing numerical measurements using mean imputation.
4. Visualize consumption trends and distributions.
5. Define High and Low labels using the median `Global_active_power`.
6. Sample 100,000 records and split them into training and testing sets.
7. Train Decision Tree and Logistic Regression classifiers.
8. Compare model performance using classification metrics.

## Preliminary Results

| Model | Test Accuracy |
|---|---:|
| Decision Tree | 99.17% |
| Logistic Regression | 99.24% |

These are results from the current experiment using a random 80/20 split of a 100,000-record sample.

The High/Low target is derived from `Global_active_power`, while the input features are related electricity measurements. Consequently, these scores should not be interpreted as evidence of future electricity forecasting or generalization to other households.

## Technology Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## Project Structure

```text
household-electricity-classification/
├── app.py
├── electricity_project.ipynb
├── requirements.txt
├── README.md
├── .gitignore
├── data/
├── models/
│   ├── logistic_regression.joblib
│   └── standard_scaler.joblib
└── assets/
    ├── daily_consumption.png
    ├── hourly_consumption.png
    ├── class_distribution.png
    └── model_comparison.png
```

## Run Locally

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Download the UCI dataset if you want to run the notebook.
5. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

## Limitations

- The High/Low labels are derived from the dataset's active-power measurements.
- Several input features are strongly related to the target.
- Evaluation uses a random split from a sample rather than a chronological future test set.
- Results do not establish that the model can forecast future electricity demand.
- The dataset represents one household, so results may not generalize to other households.

## Author

**Aman Kumar**

- GitHub: [theamankalwar](https://github.com/theamankalwar)
- LinkedIn: [Aman Kumar](https://www.linkedin.com/in/theamankalwar/)
- Email: theamankalwar@gmail.com
