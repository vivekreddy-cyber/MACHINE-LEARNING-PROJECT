# ============================================================
# LINEAR REGRESSION - PLACEMENT PREDICTION DATASET
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. FILE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\ganta vivek reddy\Videos\placement_prediction\dataset\final_preprocess_M2.csv"

OUTPUT_FOLDER = r"C:\Users\ganta vivek reddy\Videos\placement_prediction\outputs\Linear_Regression_with_Metrics_M2"

IMAGE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "images"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

# low_memory=False avoids mixed-type warning while reading
df = pd.read_csv(
    DATASET_PATH,
    low_memory=False
)

print("=" * 60)
print("LINEAR REGRESSION - PLACEMENT PREDICTION")
print("=" * 60)


print("\nDataset Shape:")
print(df.shape)


print("\nFirst 5 Records:")
print(df.head())


# ============================================================
# 3. DISPLAY COLUMN NAMES
# ============================================================

print("\nDataset Columns:")

for column in df.columns:
    print(column)


# ============================================================
# 4. SELECT FEATURES AND TARGET
# ============================================================

# Multiple Linear Regression
#
# x1 = CGPA
# x2 = AptitudeTestScore
# x3 = CodingTestScore
# x4 = MockInterviewScore
#
# y = PlacementStatus

feature_columns = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterviewScore"
]

target_column = "PlacementStatus"


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = feature_columns + [target_column]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR!")
    print("The following columns were not found:")
    print(missing_columns)

    print("\nAvailable columns are:")
    print(list(df.columns))

    raise ValueError(
        "Please change feature_columns and target_column "
        "according to your dataset."
    )


# ============================================================
# 6. CREATE MODEL DATA
# ============================================================

model_df = df[required_columns].copy()


# ============================================================
# 7. DATA CLEANING
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

print("\nData types BEFORE cleaning:")
print(model_df.dtypes)


# Convert all model columns to numeric
#
# Example:
# "85"  -> 85
# " 85 " -> 85
# " "   -> NaN
# "abc" -> NaN

for column in required_columns:

    model_df[column] = pd.to_numeric(
        model_df[column]
        .astype(str)
        .str.strip(),
        errors="coerce"
    )


print("\nData types AFTER cleaning:")
print(model_df.dtypes)


# ============================================================
# 8. CHECK MISSING / INVALID VALUES
# ============================================================

print("\nMissing / invalid values:")

missing_values = model_df.isnull().sum()

print(missing_values)


total_invalid = missing_values.sum()

print(
    "\nTotal invalid/missing values:",
    total_invalid
)


# ============================================================
# 9. REMOVE INVALID ROWS
# ============================================================

if total_invalid > 0:

    print(
        "\nRemoving rows containing "
        "missing/invalid values..."
    )

    before_rows = len(model_df)

    model_df = model_df.dropna()

    after_rows = len(model_df)

    removed_rows = before_rows - after_rows

    print("Rows before cleaning:", before_rows)
    print("Rows after cleaning :", after_rows)
    print("Rows removed        :", removed_rows)

else:

    print("\nNo missing or invalid values found.")


# ============================================================
# 10. FINAL DATA CHECK
# ============================================================

print("\nFinal Model Dataset Shape:")
print(model_df.shape)


print("\nFinal Missing Values:")
print(model_df.isnull().sum())


# Make sure there is data remaining
if model_df.empty:

    raise ValueError(
        "No valid data remains after cleaning."
    )


# ============================================================
# 11. DEFINE X AND Y
# ============================================================

X = model_df[feature_columns]

y = model_df[target_column]


print("\n" + "=" * 60)
print("FEATURES AND TARGET")
print("=" * 60)

print("\nFeatures:")
print(feature_columns)

print("\nTarget:")
print(target_column)


print("\nX Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)


# ============================================================
# 12. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 13. CREATE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 14. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

model.fit(
    X_train,
    y_train
)

print("\nModel training completed successfully.")


# ============================================================
# 15. MODEL COEFFICIENTS
# ============================================================

# Equation:
#
# y = b0 + b1x1 + b2x2 + b3x3 + b4x4

print("\nIntercept (b0):")
print(model.intercept_)


print("\nCoefficients:")

coefficient_df = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient": model.coef_
})

print(coefficient_df)


# ============================================================
# 16. LINEAR REGRESSION EQUATION
# ============================================================

equation = (
    f"{target_column} = "
    f"{model.intercept_:.4f}"
)

for feature, coefficient in zip(
    feature_columns,
    model.coef_
):

    equation += (
        f" + ({coefficient:.4f} × {feature})"
    )


print("\n" + "=" * 60)
print("LINEAR REGRESSION EQUATION")
print("=" * 60)

print("\n" + equation)


# ============================================================
# 17. PREDICTION
# ============================================================

y_pred = model.predict(X_test)


print("\nPrediction completed.")


# ============================================================
# 18. EVALUATION METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nMAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 19. CREATE PREDICTION RESULTS
# ============================================================

results = X_test.copy()

results["Actual"] = y_test.values

results["Predicted"] = y_pred

results["Residual"] = (
    results["Actual"] -
    results["Predicted"]
)

results["Absolute_Error"] = abs(
    results["Residual"]
)


print("\nFirst 10 Prediction Results:")
print(results.head(10))


# ============================================================
# 20. SAVE PREDICTION RESULTS
# ============================================================

prediction_file = os.path.join(
    OUTPUT_FOLDER,
    "linear_regression_predictions.csv"
)

results.to_csv(
    prediction_file,
    index=False
)

print("\nPrediction results saved to:")
print(prediction_file)


# ============================================================
# 21. SAVE MODEL COEFFICIENTS
# ============================================================

coefficient_file = os.path.join(
    OUTPUT_FOLDER,
    "linear_regression_coefficients.csv"
)

coefficient_df.to_csv(
    coefficient_file,
    index=False
)

print("\nModel coefficients saved to:")
print(coefficient_file)


# ============================================================
# 22. SAVE MODEL METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ],
    "Value": [
        mae,
        mse,
        rmse,
        r2
    ]
})


metrics_file = os.path.join(
    OUTPUT_FOLDER,
    "linear_regression_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)

print("\nModel metrics saved to:")
print(metrics_file)


# ============================================================
# 23. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)


# Perfect prediction line

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel(
    "Actual Placement"
)

plt.ylabel(
    "Predicted Placement"
)

plt.title(
    "Linear Regression: Actual vs Predicted Placement"
)

plt.grid(True)


actual_predicted_image = os.path.join(
    IMAGE_FOLDER,
    "actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_image,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nActual vs Predicted graph saved to:")
print(actual_predicted_image)


# ============================================================
# 24. RESIDUAL GRAPH
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_pred,
    results["Residual"],
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Placement"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Residual Plot - Linear Regression"
)

plt.grid(True)


residual_image = os.path.join(
    IMAGE_FOLDER,
    "residual_plot.png"
)

plt.savefig(
    residual_image,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nResidual graph saved to:")
print(residual_image)


# ============================================================
# 25. COEFFICIENT GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    coefficient_df["Feature"],
    coefficient_df["Coefficient"]
)

plt.xlabel(
    "Features"
)

plt.ylabel(
    "Coefficient"
)

plt.title(
    "Linear Regression Feature Coefficients"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.grid(
    axis="y"
)


coefficient_image = os.path.join(
    IMAGE_FOLDER,
    "feature_coefficients.png"
)

plt.savefig(
    coefficient_image,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nCoefficient graph saved to:")
print(coefficient_image)


# ============================================================
# 26. SAVE EQUATION
# ============================================================

equation_file = os.path.join(
    OUTPUT_FOLDER,
    "linear_regression_equation.txt"
)

with open(
    equation_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Linear Regression Equation\n"
    )

    file.write(
        "=" * 40 + "\n"
    )

    file.write(
        equation
    )


print("\nEquation saved to:")
print(equation_file)


# ============================================================
# 27. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nOutput Folder:")
print(OUTPUT_FOLDER)

print("\nGenerated Files:")

print(
    "- linear_regression_predictions.csv"
)

print(
    "- linear_regression_coefficients.csv"
)

print(
    "- linear_regression_metrics.csv"
)

print(
    "- linear_regression_equation.txt"
)

print("\nGenerated Images:")

print(
    "- actual_vs_predicted.png"
)

print(
    "- residual_plot.png"
)

print(
    "- feature_coefficients.png"
)

print("\n" + "=" * 60)