import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from .eda import DATA_DIR, DATA_FILE, FIGURES_DIR, load_data, NUMERIC_COLS, CATEGORICAL_COLS, check_missing_and_duplicates_values

PREPROCESSING_DIR = os.path.join(FIGURES_DIR, 'Preprocessing')
PROCESSED_FILE = os.path.join(DATA_DIR, 'shopping_behavior_processed.csv')


def check_unique_features(df):
    print("\nUnique value count of each feature:")
    unique_counts = df.nunique()
    for col in NUMERIC_COLS:
        print(f"- [{col}]: {unique_counts[col]}")

def detect_outliers(df: pd.DataFrame, numeric_cols: list, output_dir=None):
    print("\n=== Outlier Detection (IQR method) ===")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        print(f"- [{col}]: {len(outliers)} outliers, outlier values = {outliers.tolist()}")

    plt.figure(figsize=(15, 5))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(1, len(numeric_cols), i)
        sns.boxplot(y=df[col])
        plt.title(f"Box Plot - {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Outlier_BoxPlot_IQR.png"))
    plt.close()
    
def encode_categorical(df: pd.DataFrame):
    one_hot_cols = ['Gender', 'Category']
    df = pd.get_dummies(df, columns=one_hot_cols, prefix=one_hot_cols, dtype=int)

    label_cols = [col for col in CATEGORICAL_COLS if col not in one_hot_cols]
    
    print(f"\nList of encoder categorical variables:")
    for i, col in enumerate(label_cols):
        print(f"{i + 1}. [{col}]")

    encoding_success = True
    label_mappings = {}
    try:
        for col in label_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    except Exception as e:
        encoding_success = False
        print(f"Error in encoder process: {str(e)}")
    
    if encoding_success:
        print("=> Encoded successfully!")
        print("\n=== Label Encoding Mappings ===")
        for col, mapping in label_mappings.items():
            print(f"- {col}: \n{mapping}")
    else:
        print("=> Normalization failed. Please check the data and processing steps again.")

    return df


def check_encoded_features_correlation(df: pd.DataFrame, threshold = 0.8):
    encoded_cols = [col for col in df.columns if '_encoded' in col or 'Gender_' in col or 'Category_' in col]
    print(f"\nList of encoded features to check correlation:")
    for i, encoded_col in enumerate(encoded_cols):
        print(f"{i + 1}. [{encoded_col}]")
        
    corr = df[encoded_cols].corr()
    abs_corr = corr.abs()
    
    print(f"\nAbsolute correlation matrix:\n {abs_corr}")
    

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation of Encoded Features")
    plt.tight_layout()
    plt.savefig(os.path.join(PREPROCESSING_DIR, "Encoded_Features_Correlation.png"))
    plt.close()

    # Filter strong correlations
    high_corr = abs_corr > threshold
    np.fill_diagonal(high_corr.values, False)
    
    print("\n=== Highly Correlated Encoded Feature Pairs (>|0.8|) ===")
    to_drop = []
    for i in range(len(high_corr)):
        for j in range(i+1, len(high_corr)):
            if high_corr.iloc[i, j]:
                col_i = high_corr.index[i]
                col_j = high_corr.columns[j]
                print(f"- {col_i} ↔ {col_j}: corr = {corr.iloc[i, j]:.2f}")
                # Chọn cột thứ hai trong cặp để loại bỏ (tránh xung đột)
                if col_j not in to_drop:
                    to_drop.append(col_j)

    for col in to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            print(f"Dropped redundant column: {col}")
    return df


def normalize_numerical(df: pd.DataFrame, numeric_cols: list):
    scaler = MinMaxScaler()
    print(f"\nList of numerical features to encode:")
    for i, numeric_col in enumerate(numeric_cols):
        print(f"{i+ 1}. [{numeric_col}]")
    
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(f"{col} after MinMaxScaler")
    plt.tight_layout()
    plt.savefig(os.path.join(PREPROCESSING_DIR, "MinMaxScaler_Numerical_Histograms.png"))
    plt.close()
    print("=> Normalized successfully!")
    return df

def main():
    os.makedirs(PREPROCESSING_DIR, exist_ok=True)
    
    print("Loaded data:")
    df = load_data(DATA_FILE)
    # print(df.head())
    
    check_missing_and_duplicates_values(df)
    check_unique_features(df)
    
    detect_outliers(df, NUMERIC_COLS, PREPROCESSING_DIR)
    
    # Mã hóa đặc trưng phân loại
    df = encode_categorical(df)
    
    # Kiểm tra tương quan giữa các đặc trưng đã mã hóa
    df = check_encoded_features_correlation(df)
    
    # Chuẩn hóa đặc trưng số học
    df = normalize_numerical(df, NUMERIC_COLS)

    df.to_csv(PROCESSED_FILE, index=False)
    print(f"\nSaved processed file: [{PROCESSED_FILE}] to Data directory: [{DATA_DIR}]")

    if os.path.exists(PROCESSED_FILE):
        df_check = pd.read_csv(PROCESSED_FILE)
        print("Data just saved (first 5 rows):")
        print(df_check.head(5))
    else:
        print(f"File {PROCESSED_FILE} does not exist!")


if __name__ == '__main__':
    main()
    print(f"\n =>Preprocessing pipeline complete. Outputs saved under [ {FIGURES_DIR} ] and [ {DATA_DIR} ]")