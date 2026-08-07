import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. Load Dataset
# ===============================

print("1. Load the Dataset")

# Replace this with the actual path to your CSV file
file_path = r"C:\Users\ganta vivek reddy\Videos\placement_prediction\dataset\placement_predict_50K_Raw.csv"

try:
    # Read dataset
    df = pd.read_csv(file_path)

    # Show complete dataset
    print("-" * 60)
    print("1. Dataset Contents")
    print("-" * 60)
    print(df)

    # Dataset shape
    print("-" * 60)
    print("2. Number of Rows and Columns:", df.shape)

    # Column names
    print("-" * 60)
    print("3. Column Names")
    print(df.columns.tolist())

    # Display settings
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # First 10 rows
    print("-" * 60)
    print("4. First 10 Records")
    print(df.head(10))

    # Last 10 rows
    print("-" * 60)
    print("Last 10 Records")
    print(df.tail(10))

    # ===============================
    # Understand Dataset
    # ===============================

    print("-" * 60)
    print("5. Data Types")
    print(df.dtypes)

    print("=" * 60)

    print("6. Column Name and Data Type")
    print("-" * 40)
    for column in df.columns:
        print(f"{column:<20} {df[column].dtype}")

    print("-" * 60)
    print("7. Dataset Information")
    df.info()

    print("=" * 60)

    # Numeric columns
    print("8. Numeric Columns")

    numeric_df = df.select_dtypes(include='number')

    print(numeric_df)

    print("9. Missing Values in Numeric Columns")
    print(numeric_df.isnull().sum())

    print("Total Missing Numeric Values:",
          numeric_df.isnull().sum().sum())

    # Float columns
    float_columns = df.select_dtypes(include=['float64']).columns

    print("-" * 60)
    print("10. Float Column Names")

    for col in float_columns:
        print(col)

    print("-" * 60)
    print("11. Missing Values in Float Columns")

    print(df[float_columns].isnull().sum())

    print("Total Missing Float Values:",
          df[float_columns].isnull().sum().sum())

    # Categorical columns
    categorical_df = df.select_dtypes(include=['object'])

    print("-" * 60)
    print("12. Categorical Columns")

    print(categorical_df)

    print("Missing Values")

    print(categorical_df.isnull().sum())

    print("Total Missing Categorical Values:",
          categorical_df.isnull().sum().sum())

    # Missing values
    print("-" * 60)
    print("13. Missing Values in Each Column")

    print(df.isnull().sum())

    print("Total Missing Values:",
          df.isnull().sum().sum())

    # Duplicate records
    print("-" * 60)
    print("14. Duplicate Records")

    print(df.duplicated().sum())

    # Statistics
    print("-" * 60)
    print("15. Statistical Summary")

    print(df.describe(include='all'))

    # Histogram
    print("-" * 60)
    print("16. Histogram of CGPA")

    if "CGPA" in df.columns:
        plt.figure(figsize=(8,5))
        plt.hist(df["CGPA"].dropna(),
                 bins=10,
                 edgecolor="black")

        plt.title("Histogram of CGPA")
        plt.xlabel("CGPA")
        plt.ylabel("Frequency")
        plt.grid(True)

        plt.show()
    else:
        print("Column 'CGPA' not found in dataset.")

except FileNotFoundError:
    print("CSV file not found.")
    print(file_path)

except pd.errors.EmptyDataError:
    print("CSV file is empty.")

except pd.errors.ParserError:
    print("CSV file format is incorrect.")

except Exception as e:
    print("Error:", e)