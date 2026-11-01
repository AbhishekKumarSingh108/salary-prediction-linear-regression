"""
Salary Prediction using Simple Linear Regression
A beginner-friendly machine learning project demonstrating traditional ML workflow.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    # 1. Load Dataset
    data_path = os.path.join(os.path.dirname(__file__), "Salary_Data.csv")
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # 2. Basic Exploratory Data Analysis (EDA)
    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Dataset Summary ---")
    print(df.describe())

    # 3. Feature & Target Selection
    # X: Independent variable (Years of Experience)
    # y: Dependent variable (Salary)
    X = df[["YearsExperience"]].values
    y = df["Salary"].values

    # 4. Split dataset into Training set (80%) and Test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Train Simple Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\n--- Model Trained ---")
    print(f"Slope (Coefficient): {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Equation: Salary = {model.coef_[0]:.2f} * Experience + {model.intercept_:.2f}")

    # 6. Make Predictions on Test Set
    y_pred = model.predict(X_test)

    # Comparison DataFrame
    results_df = pd.DataFrame({
        "Actual Salary": y_test,
        "Predicted Salary": np.round(y_pred, 2),
        "Difference": np.round(y_test - y_pred, 2)
    })
    print("\n--- Test Set Predictions vs Actual ---")
    print(results_df)

    # 7. Evaluate Model Performance
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Model Evaluation ---")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")

    # 8. Visualize Results and Save Plot
    plt.figure(figsize=(9, 5))

    # Plot Training points
    plt.scatter(X_train, y_train, color="royalblue", label="Training Data", alpha=0.7)
    # Plot Test points
    plt.scatter(X_test, y_test, color="darkorange", label="Test Data (Actual)", s=70, edgecolors="black")
    # Plot Regression line
    plt.plot(X, model.predict(X), color="crimson", linewidth=2, label="Regression Line")

    plt.title("Salary vs Experience (Linear Regression)", fontsize=13)
    plt.xlabel("Years of Experience", fontsize=11)
    plt.ylabel("Salary ($)", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plot_file = os.path.join(os.path.dirname(__file__), "regression_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to: {plot_file}")

    # 9. Example Prediction for Custom Input
    custom_exp = 6.5
    predicted_val = model.predict([[custom_exp]])[0]
    print(f"\n[Sample Prediction] Estimated salary for {custom_exp} years experience: ${predicted_val:,.2f}")


if __name__ == "__main__":
    main()
