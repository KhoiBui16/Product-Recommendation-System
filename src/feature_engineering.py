import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from scipy.stats import chi2_contingency
from .preprocessing import DATA_DIR, FIGURES_DIR, load_data

PROCESSED_FILE = os.path.join(DATA_DIR, "shopping_behavior_processed.csv")
FEATURE_ENGINEERING_DIR = os.path.join(FIGURES_DIR, "Feature_Engineering")
NEW_FEAUTRES_DIR = os.path.join(FEATURE_ENGINEERING_DIR, "New_features")
FINAL_FEATURES_FILE = os.path.join(DATA_DIR, "shopping_behavior_final_features.csv")
USER_ITEM_MATRIX_FILE = os.path.join(DATA_DIR, "user_item_matrix.csv")
UNNECESSARY_FEATURES_DROPPING_DIR = os.path.join(FEATURE_ENGINEERING_DIR, "unnecessary_features_dropping")
FEATURES_SELECTION_DIR = os.path.join(FEATURE_ENGINEERING_DIR, "features_selection"
)


def create_product_features(df, verbose=False):
    """
    Create product features: Category_reconstructed, Product_ID, Product_Category.
    """

    # Reconstruct category
    categories = [
        col.replace("Category_", "")
        for col in df.columns
        if col.startswith("Category_")
    ]
    if verbose:
        print("List of one-hot categorical features:")
        for i, cat_col in enumerate(categories):
            print(f"{i + 1}. [{cat_col}]")

    def get_category(row):
        for cat in categories:
            if row.get(f"Category_{cat}", 0) == 1:
                return cat
        return "Unknown"

    df["Category_reconstructed"] = df.apply(get_category, axis=1)

    if verbose:
        print("Distribution of Category_reconstructed:")
        print(df["Category_reconstructed"].value_counts())

    # ====== Create Product_ID and Product_Category features ======
    df["Product_ID"] = (
        df["Item Purchased"]
        + "_"
        + df["Category_reconstructed"]
        + "_"
        + df["Size"]
        + "_"
        + df["Color"]
    )
    df["Product_Category"] = df["Item Purchased"] + "_" + df["Category_reconstructed"]

    if verbose:
        print("Top 5 Product_Category:")
        print(df["Product_Category"].value_counts().head())

    print(
        f"=> Created new features: [Product_Category] successfully!\n - Number of product categories: {df['Product_Category'].nunique()}"
    )
    
    print("=> Created new features: [Product_Category] successfully!")
    return df


def create_dominant_season(df):
    # Calculate the number of occurrences of each Item Purchased in each Season and store in product_season
    product_season = (
        df.groupby(["Item Purchased", "Season"]).size().unstack(fill_value=0)
    )

    # Determine the season with the highest occurrence for each Item and create `Dominant_Season`
    product_season["Dominant_Season"] = product_season.idxmax(axis=1)

    print(f"Number of unique items: {df['Item Purchased'].nunique()}")
    print(f"Number of items in product_season: {len(product_season)}")

    # Merge df_processed by `Item Purchased`
    if "Dominant_Season" not in df.columns:
        df = df.merge(
            product_season["Dominant_Season"],
            left_on="Item Purchased",  # merge key for product
            right_index=True,
            how="left",  # keep all records from df_processed
            validate="many_to_one",  # Ensure each value in "Item Purchased" maps to at most one Dominant_Season
        )

    print(f"Missing Dominant_Season after merge: {df['Dominant_Season'].isna().sum()}")
    print("Transaction counts by Dominant_Season:")
    print(df["Dominant_Season"].value_counts())

    plt.figure(figsize=(8, 6))
    sns.countplot(
        data=df,
        x="Dominant_Season",
        hue="Dominant_Season",
        palette="Set2",
        order=["Spring", "Summer", "Fall", "Winter"],
        legend=False,
    )
    plt.title("Distribution of Dominant Season", fontsize=12)
    plt.xlabel("Dominant Season", fontsize=10)
    plt.ylabel("Transaction Count", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(NEW_FEAUTRES_DIR, "dominant_season_distribution.png"))
    plt.close()

    # Encode Season and Dominant_Season using One-hot Encoding
    df = pd.get_dummies(
        df, columns=["Dominant_Season"], prefix="Dominant_Season", dtype=int
    )
    
    print("=> Created new features: [Dominant_Season] successfully!")
    return df


def create_loyalty_score(df):
    freq_map = {
        "Annually": 1,
        "Quarterly": 2,
        "Every 3 Months": 2,
        "Monthly": 3,
        "Fortnightly": 4,
        "Weekly": 5,
        "Bi-Weekly": 6,
    }
    df["Frequency_score"] = df["Frequency of Purchases"].map(freq_map)
    df["Frequency_score"] = MinMaxScaler().fit_transform(df[["Frequency_score"]])
    df["Customer_Loyalty_Score"] = (
        df["Previous Purchases"] * 0.4
        + df["Frequency_score"] * 0.4
        + df["Subscription Status_encoded"] * 0.2
    )

    plt.figure(figsize=(8, 6))
    sns.histplot(df["Customer_Loyalty_Score"], kde=True, bins=30, color="skyblue")
    plt.title("Customer Loyalty Score Distribution", fontsize=12)
    plt.xlabel("Loyalty Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(
        os.path.join(NEW_FEAUTRES_DIR, "Customer_Loyal_Score_distribution_MinMax.png")
    )
    plt.close()
    
    print("=> Created new features: [Customer_Loyalty_Score] successfully!")
    return df


def create_interaction_score(df):

    df["Interaction_Score"] = (
        df["Review Rating"] * 0.7 + df["Purchase Amount (USD)"] * 0.3
    )

    plt.figure(figsize=(8, 6))
    sns.histplot(df["Interaction_Score"], kde=True, bins=30, color="lightgreen")
    plt.title("Interaction Score Distribution", fontsize=12)
    plt.xlabel("Interaction Score", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.tight_layout()
    plt.savefig(
        os.path.join(NEW_FEAUTRES_DIR, "Interaction_Score_distribution_MinMax.png")
    )
    plt.close()
    print("=> Created new features: [Interaction_Score] successfully!")

    return df


def create_user_item_matrix(df):
    # Create `User-Item` matrix from **Interaction_Score** - **Customer ID** - **Product_Category**

    user_item_matrix = df.pivot_table(
        index="Customer ID",
        columns="Product_Category",  # Each column is Item + category
        values="Interaction_Score",  # Each cell is the interaction value for the user
        aggfunc="mean",  # If multiple interactions exist for a user & item, take the mean
    ).fillna(0)

    try:
        user_item_matrix.to_csv(USER_ITEM_MATRIX_FILE)
        print(
            f"Saved User-Item matrix: [{USER_ITEM_MATRIX_FILE}] to Data directory: [{DATA_DIR}]"
        )
    except Exception as e:
        print(f"Error saving file: {e}")

    top_N = 40
    plt.figure(figsize=(12, 8))
    sns.heatmap(user_item_matrix.iloc[:top_N], cmap="Blues", cbar=True)
    plt.title(f"Heatmap of User-Item Matrix (Top {top_N} Users)")
    plt.xlabel("Product Category")
    plt.ylabel("Customer ID")
    plt.tight_layout()
    plt.savefig(os.path.join(NEW_FEAUTRES_DIR, "Heatmap_User_Item_Matrix.png"))
    plt.close()
    
    print("=> Created new features: [user_item_matrix] successfully!")


def remove_unnecessary_features(df, threshold=1):
    print("\n===> DEBUG: feature before call remove_unnecessary_features():")
    print(sorted(df.columns.tolist()))
    print(f"All features in df before remove all unnecessary_features: {len(df.columns)}")
    
    cols_to_drop = [
        'Promo Code Used', 'Gender', 'Category', 'Location', 'Size', 'Color',
        'Subscription Status', 'Payment Method', 'Shipping Type', 'Discount Applied',
        'Frequency of Purchases', 'Frequency_score', 'Product_ID', 'Purchase Amount (USD)', 'Season'
    ]
    
    cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
    print("Unnecessary columns to be dropped:")
    for i, col in enumerate(cols_to_drop_existing):
        print(f"{i + 1}. [{col}]")

    df = df.drop(columns=cols_to_drop_existing, errors='ignore')
    print(f"Dropped {len(cols_to_drop_existing)} columns.")
    
    print("Checking features with high variance:")
    numeric_features = df.select_dtypes(include=['int32', 'int64', 'float64'])
    variances = numeric_features.var()
    variances_df = pd.DataFrame({'Feature': variances.index, 'Variance': variances.values})
    variances_df = variances_df.sort_values('Variance', ascending=False)
    
    print("\nTop 10 features with highest variance in df_final_features:")
    top_10_high_variance = variances_df.head(10)['Feature'].tolist()
    for i, col in enumerate(top_10_high_variance, 1):
        print(f"{i}. [{col}]")

    # Drop features with high variance but keep Customer ID
    high_variance_features = variances_df[
        (variances_df["Variance"] > threshold)
    ]["Feature"].tolist()
    print(f"\nFeatures with variance > {threshold}:")
    print(high_variance_features)
    
    if high_variance_features:
        plt.figure(figsize=(15, 5 * min(3, len(high_variance_features))))
        for i, feature in enumerate(high_variance_features[:3]):
            plt.subplot(min(3, len(high_variance_features)), 1, i+1)
            sns.histplot(df[feature], kde=True)
            plt.title(f'Distribution of {feature} (Variance: {variances[feature]:.6f})')
        plt.tight_layout()
        save_path = os.path.join(UNNECESSARY_FEATURES_DROPPING_DIR, "features_with_high_variaces.png")
        plt.savefig(save_path)
        plt.close()
    

    additional_cols_to_drop = [col for col in high_variance_features if not col.startswith('Category_') and col in df.columns]
    df = df.drop(columns=additional_cols_to_drop, errors='ignore')

    binary_features = [col for col in df.columns if set(df[col].dropna().unique()).issubset({0, 1})]
    print("\nBinary features in df_final_features:", binary_features)

    print("\nRatio of the most common value in binary features:")
    for col in binary_features:
        value_counts = df[col].value_counts(normalize=True)
        most_common_value = value_counts.index[0]
        most_common_ratio = value_counts.iloc[0]
        print(f"{col}: {most_common_value} ({most_common_ratio:.2%})")

    print(f"\nFeatures recommended for removal (high variance) (excluding Category_*): {additional_cols_to_drop}")

    # Drop features with high variance (protecting Category_* features)
    if additional_cols_to_drop:
        df = df.drop(columns=additional_cols_to_drop, errors='ignore')
        print(f"Dropped {len(additional_cols_to_drop)} additional features. Remaining features: {len(df.columns)}")
    else:
        print("No additional features recommended for removal.")

    print("\nRemaining columns in df_final_features:")
    for final_col in df.columns.tolist():
        print(f"[{final_col}]")
    print(f"Number of columns remaining after dropping: {len(df.columns)}")
    return df


def feature_selection(df, corr_threshold=0.8, anova_pval_thresh=0.1, chi2_pval_thresh=0.1):
    # --- 1. Based on Correlation Matrix ---
    correlation_matrix = df.corr(numeric_only=True)
    correlation_matrix_dir = os.path.join(FEATURES_SELECTION_DIR, "Correlation_Matrix")
    os.makedirs(correlation_matrix_dir, exist_ok=True)

    plt.figure(figsize=(18, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
    plt.title("Correlation Matrix of Final Features", fontsize=16)
    plt.yticks(rotation=0)
    save_path = os.path.join(correlation_matrix_dir, "Heapmap_correlation_final_features_plot.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    corr_pairs = correlation_matrix.unstack().reset_index()
    corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
    corr_pairs = corr_pairs[corr_pairs['Feature 1'] != corr_pairs['Feature 2']]
    corr_pairs['Abs Correlation'] = corr_pairs['Correlation'].abs()
    strong_corr_pairs = corr_pairs[corr_pairs['Abs Correlation'] > corr_threshold].sort_values(by='Abs Correlation', ascending=False)
    strong_corr_pairs = strong_corr_pairs.drop_duplicates(subset=['Abs Correlation'])
    
    print(f"\nStrong correlation pairs (by absolute value, threshold = {corr_threshold}):")
    print(strong_corr_pairs[['Feature 1', 'Feature 2', 'Correlation']])

    cols_to_drop_corr = []
    for _, row in strong_corr_pairs.iterrows():
        feature1, feature2 = row["Feature 1"], row["Feature 2"]
        if feature1 == "Interaction_Score":
            cols_to_drop_corr.append(feature2)
        elif feature2 == "Interaction_Score":
            cols_to_drop_corr.append(feature1)

    df = df.drop(columns=list(set(cols_to_drop_corr)), errors='ignore')
    print(f"\nFeatures recommended for removal (high correlation): {cols_to_drop_corr}")

    # --- 2. Based on Model-Based Feature Importance ---
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

    print("\nColumns in df_final_features after synchronization:")
    for col in df.columns.tolist():
        print(f"[{col}]")
    print(f"Number of columns in df_final_features: {len(df.columns)}")

    string_columns = [col for col in df.columns if df[col].dtype == 'object']
    print("\nAutomatically detected string columns:", string_columns)
    string_columns = list(set(string_columns))
    string_columns = [col for col in string_columns if col in df.columns]
    print("\nString columns to be dropped:", string_columns)

    model_features = [col for col in df.columns if col not in string_columns and col != 'Interaction_Score']
    print("Features selected for the model:", model_features)

    X = df[model_features]
    y = df['Interaction_Score']

    print("\nData types of columns in X:")
    print(X.dtypes)

    print("\nTarget variable (Interaction_Score) - First 10 rows:")
    print(y.head(10))
    
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    print("Random Forest model trained successfully.")

    importances = rf.feature_importances_
    feature_names = X.columns
    feature_importances_df = pd.DataFrame({'Features': feature_names, 'Importances': importances})
    feature_importances_df = feature_importances_df.sort_values(by='Importances', ascending=False)
    importances_pct = importances / np.sum(importances) * 100
    feature_importances_pct = pd.DataFrame({"Features": feature_names, "Importances (%)": importances_pct})
    feature_importances_pct = feature_importances_pct.sort_values(by="Importances (%)", ascending=False)
    feature_importances_pct["Cumulative Importances (%)"] = feature_importances_pct["Importances (%)"].cumsum()
    rf_features = feature_importances_pct.head(12)['Features'].tolist()

    # --- 3. Based on ANOVA F-test ---
    X_float = X.select_dtypes(include='float64')
    f_values, p_values = f_regression(X_float, y)
    anova_results = pd.DataFrame({"Features": X_float.columns, "F-Values": f_values, "p-values": p_values})
    anova_results = anova_results.sort_values(by="F-Values", ascending=False)
    significant_features_anova = anova_results[anova_results['p-values'] < anova_pval_thresh]['Features'].tolist()

    # --- 4. Based on Chi-square Test ---
    X_categories = X.select_dtypes(include=['int64', 'int32'])
    dominant_season_features = [col for col in X_categories.columns if col.startswith("Dominant_Season_")]
    dominant_season_target = df[dominant_season_features].idxmax(axis=1).str.replace('Dominant_Season_', '')

    categorical_features = [col for col in X_categories.columns if col not in dominant_season_features]
    chi2_results = {}
    for feature in categorical_features:
        contingency_table = pd.crosstab(df[feature], dominant_season_target)
        chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
        chi2_results[feature] = {"chi2": chi2_stat, "p-value": p_val}

    chi2_results_df = pd.DataFrame.from_dict(chi2_results, orient="index")
    chi2_results_df = chi2_results_df.sort_values("chi2", ascending=False)
    significant_features_chi2 = chi2_results_df[chi2_results_df['p-value'] < chi2_pval_thresh].index.tolist()

    # --- Combine all ---
    final_features = list(set(rf_features + significant_features_anova + significant_features_chi2))
    if 'Interaction_Score' not in final_features:
        final_features.append('Interaction_Score')

    print("\nList of final features after consolidation:")
    print(final_features)
    print(f"Number of final features: {len(final_features)}")

    print(f"\nList of final features in df: {df.columns.tolist()}")
    print("\n=> Finish feautre selection.")
    return df[final_features]


def save_final_features(df):
    df.to_csv(FINAL_FEATURES_FILE, index=False)
    print(f"Saved final features to: [{FINAL_FEATURES_FILE}]")


def main():
    os.makedirs(FEATURE_ENGINEERING_DIR, exist_ok=True)

    # Load preprocessed data
    df = load_data(PROCESSED_FILE)

    # Create new features
    os.makedirs(NEW_FEAUTRES_DIR, exist_ok=True)

    # Product features
    df = create_product_features(df, verbose=True)

    # Dominant_Season features
    df = create_dominant_season(df)

    # Customer_Loyalty_Score features
    df = create_loyalty_score(df)

    # Interaction_Score features
    df = create_interaction_score(df)
    
    # User_Item_Matrix
    create_user_item_matrix(df)
    
    # Remove unnecessary columns
    os.makedirs(UNNECESSARY_FEATURES_DROPPING_DIR, exist_ok=True)
    df = remove_unnecessary_features(df)
    
    # Select final features
    os.makedirs(FEATURES_SELECTION_DIR, exist_ok=True)
    df = feature_selection(df)
    
    save_final_features(df)


if __name__ == "__main__":
    main()
    print(f"\n=> Preprocessing pipeline completed. Outputs saved under [{FIGURES_DIR}] and [{DATA_DIR}]")