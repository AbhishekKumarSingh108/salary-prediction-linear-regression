# Salary Prediction using Simple Linear Regression

A beginner-friendly machine learning project implementing a simple linear regression model from scratch using `scikit-learn`, `pandas`, and `numpy`.

## 📌 Project Overview
This project predicts employee salary based on years of work experience. It covers the fundamental machine learning workflow:
- Loading and inspecting tabular data
- Splitting data into training and testing sets
- Fitting a linear regression model
- Model evaluation with standard regression metrics ($R^2$, MAE, RMSE)
- Visualizing actual vs. predicted results

## 📂 Project Structure
```text
├── Salary_Data.csv         # Dataset containing Years of Experience vs Salary
├── salary_prediction.py    # Main script: data loading, training, and evaluation
├── regression_plot.png     # Output plot showing the fitted regression line
├── requirements.txt        # Python dependencies
├── .gitignore              # Files to ignore in Git
└── README.md               # Project documentation
```

## 🛠️ Requirements & Installation
1. Clone or download this repository.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Project
Execute the main script:
```bash
python salary_prediction.py
```

## 📊 Model Evaluation Metrics
* **Equation:** $\text{Salary} = \text{Slope} \times \text{Experience} + \text{Intercept}$
* **$R^2$ Score:** Measures how well the model explains variance in the test set ($~0.90+$).
* **MAE / RMSE:** Measures average deviation of predicted salary from actual values.

## 📈 Output Plot
The script generates `regression_plot.png` displaying the data points and the fitted regression line.
