import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import kaggle
from typing import List, Tuple

# config.py
csv_file = 'shopping_behavior_updated.csv'
ROOT_DIR = os.path.dirname((__file__))
DATA_DIR = os.path.join(ROOT_DIR, ".." ,'data')
DATA_FILE = os.path.join(DATA_DIR, csv_file)
FIGURES_DIR = os.path.join(ROOT_DIR, "..", "Figures")
EDA_DIR = os.path.join(FIGURES_DIR, 'EDA')
UNIVARIATE_DIR = os.path.join(EDA_DIR, "Univariate")
UNI_NUM_DIR = os.path.join(UNIVARIATE_DIR, "Numerical")
UNI_CAT_DIR = os.path.join(UNIVARIATE_DIR, "Categorical")
BIVARIATE_DIR = os.path.join(EDA_DIR, "Bivariate")
BIV_NUM_NUM_DIR = os.path.join(BIVARIATE_DIR, "Numerical_Numerical")
BIV_NUM_CAT_DIR = os.path.join(BIVARIATE_DIR, 'Numeric_Categorical')
BIV_CAT_CAT_DIR = os.path.join(BIVARIATE_DIR, 'Categorical_Categorical')
MULTIVARIATE_DIR = os.path.join(EDA_DIR, "Multivariate")
NUMERIC_COLS = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
CATEGORICAL_COLS = [
    'Gender',
    'Item Purchased',
    'Category',
    'Location',
    'Size',
    'Color',
    'Season',
    'Subscription Status',
    'Payment Method',
    'Shipping Type',
    'Discount Applied',
    'Promo Code Used',
    'Payment Method',
    'Frequency of Purchases'
]



def download_dataset_if_needed(dataset, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"=> Created data directory at [ {data_dir} ]")
        
        # # Xac thuc API key from kaggle.json
        # kaggle.api.authenticate()
        # kaggle.api.dataset_download_files(dataset, path=data_dir, unzip=True)
        # print(f"=> Dataset downloaded and extracted to [ {data_dir} ]")
    else:
        print(f"=> Dataset already exist at [ {data_dir} ]")
        
def load_data(data_path):
    return pd.read_csv(data_path)

def overview_data(df):
    print("==Dataset overview==")
    print(f"DataFrame shape: {df.shape}")
    print("\ndf.head:")
    print(df.head())
    print("\ndf.info:")
    print(df.info())
    print("\ndf.describe:")
    print(df.describe(include='all'))
    
def check_missing_and_duplicates_values(df):
    print("==Check Missingand Duplicates Values==")
    
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    missing_info = pd.DataFrame({
        "Missing Count" : missing_values
    })
    print(missing_info)
    
    print("\nDuplicates Values:")
    print(df.duplicated())
    print(f"Total Duplicates: {df.duplicated().sum()}")
    
def plot_age(df: pd.DataFrame, out_dir: str):
    col = 'Age'
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=col, color='blue')
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{col}_box.png'))
    plt.close()

def plot_purchase_amount(df: pd.DataFrame, out_dir: str):
    col = 'Purchase Amount (USD)'
    plt.figure(figsize=(10, 6))
    sns.histplot(df, x=col, bins=30, kde=True, color='green')
    plt.title(f'Histogram of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{col}_hist.png'))
    plt.close()
    
def plot_review_rating(df: pd.DataFrame, out_dir: str):
    col = 'Review Rating'
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=col, fill=True, color='blue')
    plt.title(f'KDE Plot of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{col}_kde.png'))
    plt.close()
    
def plot_previous_purchases(df: pd.DataFrame, out_dir: str):
    col = 'Previous Purchases'
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=col, bins=30, kde=True, color='purple')
    plt.title(f'Histogram of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{col}_hist.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df, x=col, bins=30, kde=True, color='green')
    plt.title(f'Histogram of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{col}_hist.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=col, fill=True, color='red')
    plt.title(f'KDE Plot of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{col}_kde.png'))
    plt.close()
    
def plot_category_features(df, col=None, out_dir=None):
    value_counts = df[col].value_counts()
    plt.figure(figsize=(12, 5))
    sns.barplot(x=value_counts.index, y=value_counts.values, hue=value_counts.index, palette="viridis", dodge=False, legend=False)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    filename = f'{col}_plot.png'
    savefig_path = os.path.join(out_dir, filename)
    plt.savefig(savefig_path)
    plt.close()
    
def plot_bivariate_num_num(df: pd.DataFrame, out_dir: str):
    numerical_columns = df.select_dtypes(include=np.number).columns
    correlation_matrix = df[numerical_columns].corr()

    print("\n=== Pearson Correlation Matrix ===")
    print(correlation_matrix)

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Pearson Correlation Matrix of Numerical Features', fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(out_dir, 'numerical_numerical_pearson_correlation_matrix_plot.png')
    plt.savefig(save_path)
    plt.close()
    
def plot_bivariate_num_cat(df: pd.DataFrame, num_var: str, cat_var: str, out_dir: str):
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x=cat_var, y=num_var, hue=cat_var, palette='viridis', legend=False)
    plt.title(f'Boxplot: {num_var} by {cat_var}')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{num_var.replace(" ", "_")}_{cat_var.replace(" ", "_")}_box.png'))
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.violinplot(data=df, x=cat_var, y=num_var, hue=cat_var, palette='plasma', legend=False)
    plt.title(f'Violin Plot: {num_var} by {cat_var}')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{num_var.replace(" ", "_")}_{cat_var.replace(" ", "_")}_violin.png'))
    plt.close()
    
def plot_bivariate_cat_cat(df: pd.DataFrame, x_var: str, hue_var: str, out_dir: str, order = None, top_n: int = 5):
    data = df.copy()
    if top_n and x_var in df.columns:
        top_items = df[x_var].value_counts().nlargest(top_n).index
        data = df[df[x_var].isin(top_items)]

    plt.figure(figsize=(10, 6) if not top_n else (12, 8))
    ax = sns.countplot(
        data=data,
        x=x_var if not top_n else None,
        y=x_var if top_n else None,
        hue=hue_var,
        palette="pastel" if not order else "coolwarm",
        order=order if not top_n else data[x_var].value_counts().nlargest(top_n).index
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        plt.legend(title=hue_var)
    plt.title(f"Distribution of {x_var} by {hue_var}", fontsize=14)
    plt.xlabel(x_var)
    plt.ylabel("Count")
    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{x_var.replace(" ", "_")}_{hue_var.replace(" ", "_")}_plot.png')
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_multivariate_numerical_by_gender(df: pd.DataFrame, out_dir: str):
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    cols_for_pairplot = numerical_columns + ["Gender"]
    g = sns.pairplot(df[cols_for_pairplot], hue="Gender", diag_kind="kde", corner=True, palette="pastel")
    g.suptitle("Pair Plot of Numerical Variables by Gender", y=1.02, fontsize=18)
    plt.tight_layout()
    save_path = os.path.join(out_dir, "pair_plot_numerical_by_gender.png")
    g.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_multivariate_purchase_category_gender(df: pd.DataFrame, out_dir: str):
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x="Category", y="Purchase Amount (USD)", hue="Gender", palette="coolwarm")
    plt.title("Purchase Amount by Category and Gender", fontsize=16)
    plt.xlabel('Category')
    plt.ylabel('Purchase Amount (USD)')
    plt.legend(title='Gender')
    plt.tight_layout()
    save_path = os.path.join(out_dir, "box_plot_purchase_amount_by_category_gender.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def main():
    dataset_name = "zeesolver/consumer-behavior-and-shopping-habits-dataset"
    download_dataset_if_needed(dataset_name, DATA_DIR)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(EDA_DIR, exist_ok=True)
    
    # overview, checking missing and duplicates values
    df = load_data(DATA_FILE)
    overview_data(df)
    check_missing_and_duplicates_values(df)
    
    # Phân tích đơn biến (Univariate Analysis)
    os.makedirs(UNIVARIATE_DIR, exist_ok=True)
    
    # Phân tích biến số (Numerical Variales)
    os.makedirs(UNI_NUM_DIR, exist_ok=True)
    print(f"\nColumns numerical variabls to plot: {NUMERIC_COLS}")
    plot_age(df, UNI_NUM_DIR)
    plot_purchase_amount(df, UNI_NUM_DIR)
    plot_review_rating(df, UNI_NUM_DIR)
    plot_previous_purchases(df, UNI_NUM_DIR)
    
    # Phân tích biến phân loại (Categorical Variables)
    os.makedirs(UNI_CAT_DIR, exist_ok=True)
    print(f"\nColumns categorical variables to plot: {CATEGORICAL_COLS}")
    for cat_col in CATEGORICAL_COLS:
        plot_category_features(df, cat_col, UNI_CAT_DIR)
    
    # Phân tích hai biến (Bivariate Analysis)
    os.makedirs(BIVARIATE_DIR, exist_ok=True)
    
    # Mối quan hệ giữa các biến số (Feature-Feature: Numerical)
    os.makedirs(BIV_NUM_NUM_DIR, exist_ok=True)
    plot_bivariate_num_num(df, BIV_NUM_NUM_DIR)
    
    # Mối quan hệ giữa biến số và biến phân loại (Feature-Feature/Label: Numerical vs Categorical)
    os.makedirs(BIV_NUM_CAT_DIR, exist_ok=True)
    num_cat_pairs = [
        ("Purchase Amount (USD)", "Category"),
        ("Purchase Amount (USD)", "Gender"),
        ("Purchase Amount (USD)", "Subscription Status"),
        ("Purchase Amount (USD)", "Season"),
        ("Purchase Amount (USD)", "Frequency of Purchases"),
        ("Review Rating", "Category"),
        ("Review Rating", "Subscription Status"),
        ("Age", "Gender"),
        ("Age", "Frequency of Purchases"),
        ("Previous Purchases", "Category")
    ]
    for num_var, cat_var in num_cat_pairs:
        plot_bivariate_num_cat(df, num_var, cat_var, BIV_NUM_CAT_DIR)
        
    # Mối quan hệ giữa các biến phân loại (Feature-Feature: Categorical vs Categorical)
    os.makedirs(BIV_CAT_CAT_DIR, exist_ok=True)
    
    cat_cat_pairs = [
        ("Category", "Gender", None, None),
        ("Season", "Subscription Status", ["Spring", "Summer", "Fall", "Winter"], None),
        ("Discount Applied", "Promo Code Used", None, None),
        ("Item Purchased", "Gender", None, 10),
        ("Gender", "Size", None, None),
        ("Discount Applied", "Gender", None, None)
    ]
    for x_var, hue_var, order, top_n in cat_cat_pairs:
        plot_bivariate_cat_cat(df, x_var, hue_var, BIV_CAT_CAT_DIR, order, top_n)

    # Phân tích đa biến (Multivariate Analysis)
    os.makedirs(MULTIVARIATE_DIR, exist_ok=True)

    plot_multivariate_numerical_by_gender(df, MULTIVARIATE_DIR)
    plot_multivariate_purchase_category_gender(df, MULTIVARIATE_DIR)
    
if __name__ == '__main__':
    main()
    print(f"\n=> EDA pipeline complete. Outputs saved under [ {FIGURES_DIR} ]")
