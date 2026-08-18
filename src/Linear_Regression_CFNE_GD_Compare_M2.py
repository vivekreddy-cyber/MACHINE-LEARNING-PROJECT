# ============================================================
# LINEAR REGRESSION
# Closed-Form Normal Equation vs Gradient Descent
#
# Images are stored in ONE separate folder
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = (
    r"C:\Users\ganta vivek reddy\Videos\placement_prediction"
    r"\dataset\final_preprocess_M2.csv"
)

data = pd.read_csv(
    DATASET_PATH,
    low_memory=False
)

print("=" * 60)
print("LINEAR REGRESSION")
print("CLOSED-FORM NORMAL EQUATION VS GRADIENT DESCENT")
print("=" * 60)

print("\nOriginal Dataset Shape:")
print(data.shape)

print("\nFirst 5 Records:")
print(data.head())

print("\nDataset Columns:")
print(list(data.columns))


# ============================================================
# 2. DATA CLEANING
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Remove unnecessary spaces from column names

data.columns = (
    data.columns
    .astype(str)
    .str.strip()
)

# Convert every column to numeric
#
# Valid numbers remain numbers.
# Blank spaces and invalid values become NaN.

for column in data.columns:

    data[column] = pd.to_numeric(
        data[column]
        .astype(str)
        .str.strip(),
        errors="coerce"
    )


print("\nMissing / Invalid Values:")
print(data.isnull().sum())


# Remove rows containing missing values

rows_before = len(data)

data = data.dropna()

rows_after = len(data)

rows_removed = rows_before - rows_after


print("\nRows before cleaning :", rows_before)
print("Rows after cleaning  :", rows_after)
print("Rows removed         :", rows_removed)


if data.empty:

    raise ValueError(
        "No valid data remains after cleaning."
    )


# ============================================================
# 3. EXTRACT FEATURES AND TARGET
# ============================================================

# All columns except the last column = features
# Last column = target

X = data.iloc[:, :-1].values

y = data.iloc[:, -1].values


print("\n" + "=" * 60)
print("FEATURES AND TARGET")
print("=" * 60)

print("\nNumber of Features:")
print(X.shape[1])

print("\nNumber of Samples:")
print(X.shape[0])

print("\nTarget Column:")
print(data.columns[-1])

print("\nX Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)


# ============================================================
# 4. CREATE IMAGE OUTPUT FOLDER
# ============================================================

IMAGE_FOLDER = (
    r"C:\Users\ganta vivek reddy\Videos\placement_prediction"
    r"\outputs\Linear_Regression_CFNE_GD_Compare_M2"
)

os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)

print("\nImage output folder:")
print(IMAGE_FOLDER)


# ============================================================
# 5. TRAIN-TEST SPLIT
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

print("\nTraining samples:")
print(len(X_train))

print("\nTesting samples:")
print(len(X_test))


# ============================================================
# 6. FEATURE SCALING
#    Used for Gradient Descent
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 7. CLOSED FORM SOLUTION
#    NORMAL EQUATION
# ============================================================

print("\n" + "=" * 60)
print("CLOSED FORM NORMAL EQUATION")
print("=" * 60)


# ------------------------------------------------------------
# Add bias/intercept column
# ------------------------------------------------------------

X_train_bias = np.c_[
    np.ones(
        (X_train.shape[0], 1)
    ),
    X_train
]

X_test_bias = np.c_[
    np.ones(
        (X_test.shape[0], 1)
    ),
    X_test
]


# ------------------------------------------------------------
# Normal Equation
#
# theta = (X^T X)^(-1) X^T y
#
# Using pseudo-inverse instead of direct inverse
# makes the calculation more numerically stable.
# ------------------------------------------------------------

theta_normal = (
    np.linalg.pinv(
        X_train_bias
    ).dot(
        y_train
    )
)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

pred_normal = (
    X_test_bias.dot(
        theta_normal
    )
)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

mse_normal = mean_squared_error(
    y_test,
    pred_normal
)

r2_normal = r2_score(
    y_test,
    pred_normal
)


print("\nNormal Equation Coefficients:")

print(theta_normal)


print("\nNormal Equation MSE:")
print(mse_normal)


print("\nNormal Equation R2 Score:")
print(r2_normal)


# ============================================================
# 8. GRADIENT DESCENT
# ============================================================

print("\n" + "=" * 60)
print("GRADIENT DESCENT")
print("=" * 60)


# ------------------------------------------------------------
# Add bias column to scaled data
# ------------------------------------------------------------

X_train_gd = np.c_[
    np.ones(
        (X_train_scaled.shape[0], 1)
    ),
    X_train_scaled
]

X_test_gd = np.c_[
    np.ones(
        (X_test_scaled.shape[0], 1)
    ),
    X_test_scaled
]


# ------------------------------------------------------------
# Number of training samples
# ------------------------------------------------------------

m = len(y_train)


# ------------------------------------------------------------
# Initialize theta
# ------------------------------------------------------------

theta_gd = np.zeros(
    X_train_gd.shape[1]
)


# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------

learning_rate = 0.01

epochs = 1000


# ============================================================
# 9. STORE LOSS FOR EACH EPOCH
# ============================================================

loss_history = []


# ============================================================
# 10. GRADIENT DESCENT ITERATIONS
# ============================================================

for epoch in range(epochs):

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = (
        X_train_gd.dot(
            theta_gd
        )
    )


    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    errors = (
        predictions -
        y_train
    )


    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    gradients = (
        (2 / m)
        *
        X_train_gd.T.dot(
            errors
        )
    )


    # --------------------------------------------------------
    # Update parameters
    # --------------------------------------------------------

    theta_gd -= (
        learning_rate *
        gradients
    )


    # --------------------------------------------------------
    # Calculate training MSE
    # --------------------------------------------------------

    loss = np.mean(
        errors ** 2
    )

    loss_history.append(
        loss
    )


# ============================================================
# 11. GRADIENT DESCENT PREDICTION
# ============================================================

pred_gd = (
    X_test_gd.dot(
        theta_gd
    )
)


# ============================================================
# 12. GRADIENT DESCENT METRICS
# ============================================================

mse_gd = mean_squared_error(
    y_test,
    pred_gd
)

r2_gd = r2_score(
    y_test,
    pred_gd
)


print("\nGradient Descent Coefficients:")

print(theta_gd)


print("\nGradient Descent MSE:")
print(mse_gd)


print("\nGradient Descent R2 Score:")
print(r2_gd)


# ============================================================
# 13. COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("NORMAL EQUATION VS GRADIENT DESCENT")
print("=" * 60)


comparison_df = pd.DataFrame({

    "Method": [
        "Normal Equation",
        "Gradient Descent"
    ],

    "MSE": [
        mse_normal,
        mse_gd
    ],

    "R2": [
        r2_normal,
        r2_gd
    ]
})


print("\n")
print(comparison_df)


# ============================================================
# 14. SAVE COMPARISON RESULTS
# ============================================================

comparison_file = os.path.join(
    IMAGE_FOLDER,
    "normal_equation_vs_gradient_descent.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False
)

print("\nComparison results saved:")
print(comparison_file)


# ============================================================
# IMAGE 1
# ACTUAL VS PREDICTED VALUES
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    y_test,
    pred_normal,
    alpha=0.5,
    label="Normal Equation"
)


plt.scatter(
    y_test,
    pred_gd,
    alpha=0.5,
    label="Gradient Descent"
)


# ------------------------------------------------------------
# Perfect prediction line
# ------------------------------------------------------------

minimum = min(
    y_test.min(),
    pred_normal.min(),
    pred_gd.min()
)

maximum = max(
    y_test.max(),
    pred_normal.max(),
    pred_gd.max()
)


plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)


plt.xlabel(
    "Actual Values"
)

plt.ylabel(
    "Predicted Values"
)

plt.title(
    "Actual vs Predicted Values"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


image1 = os.path.join(
    IMAGE_FOLDER,
    "actual_vs_predicted.png"
)


plt.savefig(
    image1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nImage 1 saved:")
print(image1)


# ============================================================
# IMAGE 2
# RESIDUAL COMPARISON
# ============================================================

# Residual = Actual - Predicted

normal_residuals = (
    y_test -
    pred_normal
)

gd_residuals = (
    y_test -
    pred_gd
)


plt.figure(
    figsize=(9, 6)
)


plt.scatter(
    pred_normal,
    normal_residuals,
    alpha=0.5,
    label="Normal Equation"
)


plt.scatter(
    pred_gd,
    gd_residuals,
    alpha=0.5,
    label="Gradient Descent"
)


plt.axhline(
    y=0,
    linestyle="--"
)


plt.xlabel(
    "Predicted Values"
)

plt.ylabel(
    "Residuals"
)

plt.title(
    "Residual Comparison"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


image2 = os.path.join(
    IMAGE_FOLDER,
    "residual_comparison.png"
)


plt.savefig(
    image2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nImage 2 saved:")
print(image2)


# ============================================================
# IMAGE 3
# GRADIENT DESCENT LOSS CURVE
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    range(
        1,
        epochs + 1
    ),
    loss_history
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Mean Squared Error"
)

plt.title(
    "Gradient Descent Convergence"
)

plt.grid(True)

plt.tight_layout()


image3 = os.path.join(
    IMAGE_FOLDER,
    "gradient_descent_loss.png"
)


plt.savefig(
    image3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nImage 3 saved:")
print(image3)


# ============================================================
# 15. SAVE IMAGE INFORMATION
# ============================================================

image_info = pd.DataFrame({

    "Image": [
        "actual_vs_predicted.png",
        "residual_comparison.png",
        "gradient_descent_loss.png"
    ],

    "Description": [
        "Actual values versus predictions from both methods",
        "Residual comparison between Normal Equation and Gradient Descent",
        "MSE loss across Gradient Descent epochs"
    ]
})


image_info_file = os.path.join(
    IMAGE_FOLDER,
    "image_information.csv"
)


image_info.to_csv(
    image_info_file,
    index=False
)


print("\nImage information saved:")
print(image_info_file)


# ============================================================
# 16. SAVE COEFFICIENTS
# ============================================================

feature_names = list(
    data.columns[:-1]
)


coefficient_df = pd.DataFrame({

    "Feature": [
        "Intercept"
    ] + feature_names,

    "Normal_Equation": theta_normal,

    "Gradient_Descent_Scaled": theta_gd
})


coefficient_file = os.path.join(
    IMAGE_FOLDER,
    "linear_regression_coefficients.csv"
)


coefficient_df.to_csv(
    coefficient_file,
    index=False
)


print("\nCoefficient information saved:")
print(coefficient_file)


# ============================================================
# 17. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 60)


print("\nAll images are stored in ONE folder:")

print(IMAGE_FOLDER)


print("\nGenerated files:")

print("1. actual_vs_predicted.png")

print("2. residual_comparison.png")

print("3. gradient_descent_loss.png")

print("4. image_information.csv")

print("5. normal_equation_vs_gradient_descent.csv")

print("6. linear_regression_coefficients.csv")


print("\nOriginal dataset was NOT modified.")

print("\n" + "=" * 60)