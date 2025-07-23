# src/training.py
import os
import time
import psutil
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, BaggingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, ndcg_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy import sparse
import optuna
from .preprocessing import DATA_DIR, FIGURES_DIR
from .feature_engineering import USER_ITEM_MATRIX_FILE, FINAL_FEATURES_FILE

# Define directories
TRAINING_DIR = os.path.join(FIGURES_DIR, "Training")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "Models")

def create_training_dir():
    """Create training directory if it doesn't exist."""
    os.makedirs(TRAINING_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Training_dir path: [ {TRAINING_DIR} ]")
    print("=> Finished create training dir")

def load_processed_data():
    """Load processed data from preprocessing and feature engineering steps."""
    final_features_csv = os.path.join(DATA_DIR, "shopping_behavior_final_features.csv")
    user_item_matrix_csv = os.path.join(DATA_DIR, "user_item_matrix.csv")
    processed_features_csv = os.path.join(DATA_DIR, "shopping_behavior_processed.csv")

    features_df = pd.read_csv(final_features_csv)
    user_item_matrix = pd.read_csv(user_item_matrix_csv, index_col=0)
    processed_features_df = pd.read_csv(processed_features_csv)
    
    print("user_item_matrix shape:", user_item_matrix.shape)
    print("users count:", user_item_matrix.shape[0])
    print("item (features) count:", user_item_matrix.shape[1])
    
    unique_customer_ids = user_item_matrix.index.nunique()
    num_rows = user_item_matrix.shape[0]
    print("Customer ID unique count:", unique_customer_ids)
    print("(rows):", num_rows)
    
    if unique_customer_ids == num_rows:
        print("OK: Each row is one unique Customer ID.")
    else:
        print(f"Warning: Having {num_rows - unique_customer_ids} rows: duplicated Customer ID!")
    
    print(f"\nUser item matrix features: {user_item_matrix.columns.tolist()}")
    print("=> Finished load_processed_data")
    return features_df, user_item_matrix, processed_features_df

def validate_data(features_df, processed_features_df, user_item_matrix):
    """Validate required columns and data consistency."""
    required_cols = [
        'Previous_Purchases', 'Gender_Female', 'Customer_Loyalty_Score', 'Size_encoded',
        'Dominant_Season_Spring', 'Category_Footwear', 'Age', 'Dominant_Season_Winter',
        'Dominant_Season_Summer', 'Discount_Applied_encoded', 'Dominant_Season_Fall',
        'Category_Clothing', 'Category_Outerwear', 'Category_Accessories', 'Interaction_Score'
    ]
    
    if not all(col in features_df.columns for col in required_cols):
        raise ValueError(f"Didn't have features: {set(required_cols) - set(features_df.columns)}")
    
    numerical_cols = ['Age', 'Previous_Purchases', 'Customer_Loyalty_Score']
    if not all(col in features_df.columns for col in numerical_cols):
        raise ValueError(f"Didn't have numeric columns: {set(numerical_cols) - set(features_df.columns)}")
    
    if 'Interaction_Score' not in features_df.columns:
        raise ValueError("Didn't have 'Interaction_Score' features in features_df")
    
    customer_id_col = 'Customer ID'
    if 'Customer ID' not in processed_features_df.columns:
        if 'Customer_ID' in processed_features_df.columns:
            customer_id_col = 'Customer_ID'
        else:
            raise ValueError("processed_features_df does not have 'Customer ID' or 'Customer_ID' feature")
    
    if 'Customer ID' not in features_df.columns:
        features_df.insert(0, 'Customer ID', processed_features_df[customer_id_col].iloc[:len(features_df)])
    
    features_df['Customer ID'] = features_df['Customer ID'].astype(int)
    user_item_matrix.index = user_item_matrix.index.astype(int)
    
    updated_features_path = os.path.join(DATA_DIR, 'shopping_behavior_final_features_with_customer_id.csv')
    features_df.to_csv(updated_features_path, index=False)
    
    print("=> Finished validate_data")
    return features_df, user_item_matrix

def split_train_test(features_df):
    """Split data into train and test sets."""
    train_df, test_df = train_test_split(features_df, test_size=0.2, random_state=42, shuffle=True)
    print(f"\ntrain_df.shape: {train_df.shape}")
    print(f"test_df.shape: {test_df.shape}")
    print("Finished split_train_test")
    return train_df, test_df

def define_user_item_features(features_df):
    """Define user and item feature columns."""
    category_cols = [col for col in features_df.columns if col.startswith("Category_")]
    season_cols = [col for col in features_df.columns if col.startswith("Dominant_Season_")]
    size_encoded_col = "Size_encoded"
    discount_encoded_col = "Discount_Applied_encoded"
    gender_female_col = "Gender_Female"
    age_col = "Age"
    previous_purchases_col = "Previous_Purchases"
    loyalty_score_col = "Customer_Loyalty_Score"
    interaction_score_col = "Interaction_Score"
    
    user_profile_features = (
        category_cols
        + season_cols
        + [size_encoded_col, discount_encoded_col, gender_female_col, age_col, previous_purchases_col, loyalty_score_col]
    )
    
    print(f"\nuser_profile_features: {user_profile_features}")
    print("=> Finished define_user_item_features")
    return user_profile_features, category_cols, season_cols, age_col, previous_purchases_col, loyalty_score_col, interaction_score_col

def validate_item_categories(features_df, user_item_matrix, category_cols):
    """Validate item categories consistency."""
    items = user_item_matrix.columns.tolist()
    categories = set(col.replace('Category_', '') for col in category_cols)
    item_categories = set(item.split('_')[-1] for item in items)
    
    if not categories.issuperset(item_categories):
        raise ValueError(f"Some categories in user_item_matrix do not match Category_*: {item_categories - categories}")
    
    print(f"\nItem categories: {item_categories}")
    print("=> Finished validate_item_categories")

def assign_item_column(df, user_item_matrix, features_df):
    """Assign items to users based on loyalty score."""
    items = []
    for cid in df['Customer ID']:
        if cid in user_item_matrix.index:
            user_row = user_item_matrix.loc[cid]
            interacted_items = user_row[user_row > 0].index.tolist()
            
            if not interacted_items:
                items.append([np.nan])
            else:
                user_data = features_df[features_df['Customer ID'] == cid]
                
                if not user_data.empty:
                    loyalty_score = user_data['Customer_Loyalty_Score'].iloc[0]
                    item_scores = {}
                    
                    for item in interacted_items:
                        category = item.split('_')[-1]
                        category_col = f'Category_{category}'
                        
                        if category_col in user_data.columns:
                            base_score = user_row[item] if not np.isnan(user_row[item]) else 0
                            item_scores[item] = base_score * loyalty_score
                    
                    if item_scores:
                        top_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                        items.append([item for item, score in top_items])
                    else:
                        items.append([np.nan])
                else:
                    items.append([np.nan])
        else:
            items.append([np.nan])
            
    print("=> Finished assign_item_columns.")
    return items

def create_user_profiles(features_df, user_profile_features, weight_col='Customer_Loyalty_Score'):
    """Create weighted user profiles."""
    if not all(col in features_df.columns for col in user_profile_features):
        raise ValueError(f"Missing user_profile_features in features_df: {set(user_profile_features) - set(features_df.columns)}")
    
    def _weighted_avg(x):
        w = x[weight_col].fillna(0)
        w_sum = w.sum()
        if not np.isscalar(w_sum):
            raise ValueError(f"w_sum is not a scalar: {w_sum}")
        if np.isclose(w_sum, 0):
            return x[user_profile_features].mean().values
        return np.average(x[user_profile_features], axis=0, weights=w)
    
    if weight_col not in features_df.columns:
        raise ValueError(f"Feature '{weight_col}' is not exists in DataFrame.")
    
    columns_to_select = list(set(user_profile_features + [weight_col]))
    result = features_df.groupby('Customer ID')[columns_to_select].apply(_weighted_avg)
    user_profiles = pd.DataFrame(result.tolist(), index=result.index, columns=user_profile_features)
    print("\nuser_profiles.columns:", user_profiles.columns)
    print("=> Finished create_user_profiles")
    return user_profiles

def create_item_features(train_df, user_item_matrix, user_profile_features):
    """Create item features."""
    if not all(col in train_df.columns for col in user_profile_features):
        raise ValueError(f"Missing user_profile_features in train_df: {set(user_profile_features) - set(train_df.columns)}")
    
    item_features = pd.DataFrame(index=user_item_matrix.columns)
    for item in user_item_matrix.columns:
        item_mask = train_df['Item'].apply(lambda x: item in x if isinstance(x, list) else False)
        if item_mask.any():
            item_df = train_df[item_mask]
            item_features.loc[item, user_profile_features] = item_df[user_profile_features].mean()
        else:
            category = item.split('_')[-1]
            category_col = f'Category_{category}'
            mask = train_df[category_col] == 1
            if mask.any():
                item_features.loc[item, user_profile_features] = train_df.loc[mask, user_profile_features].mean()
            else:
                item_features.loc[item, user_profile_features] = 0.1
    print("=> Finished create_item_features")
    return item_features.fillna(0.1)

def save_pipeline_data(user_item_matrix, user_profiles, item_features):
    """Save pipeline data for deployment."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(user_item_matrix, os.path.join(MODELS_DIR, "user_item_matrix.pkl"))
    joblib.dump(user_profiles, os.path.join(MODELS_DIR, "user_profiles.pkl"))
    joblib.dump(item_features, os.path.join(MODELS_DIR, "item_features.pkl"))
    print(f"Saved user_item_matrix to {MODELS_DIR}/user_item_matrix.pkl")
    print(f"Saved user_profiles to {MODELS_DIR}/user_profiles.pkl")
    print(f"Saved item_features to {MODELS_DIR}/item_features.pkl")
    print("=> Finished save_pipeline_data")

def prepare_user_item_matrix(train_df, test_df, user_item_matrix):
    """Prepare user-item matrices for training and testing."""
    user_item_matrix_cf = user_item_matrix.copy()
    print("\nUsing original user-item matrix because each user just interacts with each item.")
    print("\nuser_item_matrix_cf.shape:", user_item_matrix_cf.shape)
    
    print("=> Finished prepare_user_item_matrix")
    return user_item_matrix_cf

def predict_cb_score(user_id, item_id, user_profiles, item_features, category_cols, season_cols, age_col, previous_purchases_col, loyalty_score_col, train_df, min_similarity=0.15):
    """Predict content-based score with improved cold start handling."""
    if user_id not in user_profiles.index:
        category = item_id.split('_')[-1]
        category_col = f'Category_{category}'
        if category_col in train_df.columns:
            mean_score = train_df[train_df[category_col] == 1]['Customer_Loyalty_Score'].mean()
            return mean_score if not np.isnan(mean_score) else train_df['Customer_Loyalty_Score'].mean()
        return train_df['Customer_Loyalty_Score'].mean()
    
    if item_id not in item_features.index:
        return train_df['Customer_Loyalty_Score'].mean()
    
    cat_features = category_cols + season_cols
    user_cat_vec = user_profiles.loc[user_id, cat_features].values.reshape(1, -1)
    item_cat_vec = item_features.loc[item_id, cat_features].values.reshape(1, -1)
    cat_similarity = cosine_similarity(user_cat_vec, item_cat_vec)[0][0]
    
    num_features = [age_col, previous_purchases_col, loyalty_score_col]
    user_num_vec = user_profiles.loc[user_id, num_features].values
    item_num_vec = item_features.loc[item_id, num_features].values
    weights = np.array([1.0, 1.5, 2.0])
    user_weighted = user_num_vec * weights
    item_weighted = item_num_vec * weights
    num_similarity = cosine_similarity(user_weighted.reshape(1, -1), item_weighted.reshape(1, -1))[0][0]
    
    final_similarity = cat_similarity * 0.5 + num_similarity * 0.5
    score = (final_similarity + 1) / 2
    return min(max(score, 0), 1.0) if score >= min_similarity else train_df['Customer_Loyalty_Score'].mean()

class CBModel:
    """Content-Based Model."""
    def __init__(self, user_profiles, item_features, category_cols, season_cols, age_col, previous_purchases_col, loyalty_score_col, train_df):
        self.user_profiles = user_profiles
        self.item_features = item_features
        self.category_cols = category_cols
        self.season_cols = season_cols
        self.age_col = age_col
        self.previous_purchases_col = previous_purchases_col
        self.loyalty_score_col = loyalty_score_col
        self.train_df = train_df
    
    def predict(self, user_id, target_items):
        preds = []
        for item_id in target_items:
            score = predict_cb_score(
                user_id, item_id, self.user_profiles, self.item_features,
                self.category_cols, self.season_cols, self.age_col,
                self.previous_purchases_col, self.loyalty_score_col, self.train_df
            )
            preds.append(score)
        return np.array(preds)

def predict_item_cf_score(user_id, item_id, user_item_matrix, global_mean=None):
    """Predict item-item collaborative filtering score, using test matrix if provided."""
    if user_id not in user_item_matrix.index or item_id not in user_item_matrix.columns:
        return 0.1 if global_mean is None else float(global_mean)
    
    sparse_matrix = sparse.csr_matrix(user_item_matrix.values)
    item_item_matrix = sparse_matrix.T
    item_indices = user_item_matrix.columns
    item_idx = item_indices.get_loc(item_id)
    
    knn = NearestNeighbors(n_neighbors=len(item_indices), metric='cosine', algorithm='brute')
    knn.fit(item_item_matrix)
    item_vector = item_item_matrix[item_idx:item_idx+1]
    distances, indices = knn.kneighbors(item_vector, n_neighbors=len(item_indices))
    similarities = 1 - distances[0]
    similar_item_ids = item_indices[indices[0]]
    
    user_vector = user_item_matrix.loc[user_id].values
    interacted_items = [(item, score) for item, score in zip(user_item_matrix.columns, user_vector) if score > 0]
    
    if not interacted_items:
        return 0.1 if global_mean is None else float(global_mean)
    
    weighted_sum = 0
    similarity_sum = 0
    for sim_item_id, similarity in zip(similar_item_ids, similarities):
        for interacted_item_id, user_interaction in interacted_items:
            if sim_item_id == interacted_item_id:
                weighted_sum += similarity * user_interaction
                similarity_sum += abs(similarity)
                break
    
    if similarity_sum == 0:
        return 0.1 if global_mean is None else float(global_mean)
    
    score = weighted_sum / similarity_sum
    return float(np.clip(score, 0, 1))

class ItemCFModel:
    """Item-Item Collaborative Filtering Model."""
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.global_mean = user_item_matrix.values.mean()
    
    def predict(self, user_id, target_items):
        preds = []
        for item_id in target_items:
            score = predict_item_cf_score(user_id, item_id, self.user_item_matrix, self.global_mean)
            preds.append(score)
        return np.array(preds)

def hybrid_score(user_id, item_id, user_profiles, item_features, user_item_matrix, alpha=0.5, beta=0.5, category_cols=None, season_cols=None, age_col=None, previous_purchases_col=None, loyalty_score_col=None, train_df=None):
    """Calculate hybrid score combining CB and CF, using test matrix if provided."""
    cb_score = predict_cb_score(user_id, item_id, user_profiles, item_features, category_cols, season_cols, age_col, previous_purchases_col, loyalty_score_col, train_df)
    item_cf_score = predict_item_cf_score(user_id, item_id, user_item_matrix)
    return alpha * cb_score + beta * item_cf_score

class HybridModel:
    """Hybrid Model combining CB and CF."""
    def __init__(self, user_profiles, item_features, user_item_matrix, alpha=0.5, beta=0.5, category_cols=None, season_cols=None, age_col=None, previous_purchases_col=None, loyalty_score_col=None, train_df=None):
        self.user_profiles = user_profiles
        self.item_features = item_features
        self.user_item_matrix = user_item_matrix
        self.alpha = alpha
        self.beta = beta
        self.category_cols = category_cols
        self.season_cols = season_cols
        self.age_col = age_col
        self.previous_purchases_col = previous_purchases_col
        self.loyalty_score_col = loyalty_score_col
        self.train_df = train_df
    
    def predict(self, user_id, target_items):
        preds = []
        for item_id in target_items:
            score = hybrid_score(
                user_id, item_id, self.user_profiles, self.item_features, self.user_item_matrix, self.alpha, self.beta, self.category_cols, self.season_cols, self.age_col,
                self.previous_purchases_col, self.loyalty_score_col, self.train_df
            )
            preds.append(score)
        return np.array(preds)

def evaluate_models(user_id, test_df, user_profiles, all_items, models):
    """Evaluate models for a given user."""
    gt_items = test_df[test_df['Customer ID'] == user_id]['Item'].iloc[0] if user_id in test_df['Customer ID'].values else []
    gt_items = gt_items if isinstance(gt_items, list) else [gt_items] if pd.notna(gt_items) else []
    summary = {'Customer ID': user_id}
    for name, model in models:
        preds = model.predict(user_id, all_items)
        gt_scores = {item_id: round(preds[all_items.index(item_id)], 4) for item_id in gt_items if item_id in all_items}
        summary[name] = (gt_items, gt_scores)
    return summary

def print_top_n_recommendations(user_id, test_df, user_profiles, all_items, models, top_n=5):
    """Print top-N recommendations for a user."""
    print(f"\nUser {user_id}:")
    gt_items = test_df[test_df['Customer ID'] == user_id]['Item'].iloc[0] if user_id in test_df['Customer ID'].values else []
    gt_items = gt_items if isinstance(gt_items, list) else [gt_items] if pd.notna(gt_items) else []
    
    print(f"  Đã mua: {gt_items}")
    for name, model in models:
        preds = model.predict(user_id, all_items)
        top_n_idx = np.argsort(preds)[::-1][:top_n]
        top_n_items = [all_items[i] for i in top_n_idx]
        top_n_scores = [preds[i] for i in top_n_idx]
        print(f"  {name} gợi ý Top-{top_n}:")
        for rank, (item, score) in enumerate(zip(top_n_items, top_n_scores), 1):
            print(f"    {rank}. {item:25} | Score: {score:.4f}")

def create_ml_dataset(interaction_df, user_profiles, item_features, interaction_df_source, is_train=True):
    """Create dataset for ML models."""
    data = []
    for _, row in interaction_df.iterrows():
        user_id = row['Customer ID']
        item = row['Item'][0] if isinstance(row['Item'], list) else row['Item']
        
        # Tương tác thực
        user_feat = user_profiles.loc[user_id] if user_id in user_profiles.index else user_profiles.mean()
        # item_feat = item_features.loc[item] if item in item_features.index else pd.Series(0, index=item_features.columns)
        
        # bổ sung thêm 2 dòng item_feateva2 interaction_score
        item_feat = item_features.loc[item] if item in item_features.index else item_features.mean()
        interaction_score = row['Interaction_Score']
        
        
        # interaction_score = interaction_df_source[interaction_df_source['Customer ID'] == user_id]['Interaction_Score'].iloc[0] if user_id in interaction_df_source['Customer ID'].values else 0
        
        combined = pd.concat([user_feat.rename(lambda x: f"{x}_user"), item_feat.rename(lambda x: f"{x}_item")])
        combined['Customer ID'] = user_id
        combined['Item'] = item
        combined['Interaction_Score'] = interaction_score
        data.append(combined)
        
        # bổ sung thêm:
        # Tương tác giả định (negative samples)
        if is_train:
            negative_items = np.random.choice(item_features.index, size=3, replace=False)
            for neg_item in negative_items:
                if neg_item != item:
                    neg_item_feat = item_features.loc[neg_item]
                    neg_combined = pd.concat([user_feat.rename(lambda x: f"{x}_user"), neg_item_feat.rename(lambda x: f"{x}_item")])
                    neg_combined['Customer ID'] = user_id
                    neg_combined['Item'] = neg_item
                    neg_combined['Interaction_Score'] = 0  # Gán score 0 cho negative samples
                    data.append(neg_combined)
        
    
    merged = pd.DataFrame(data)
    merged = merged.fillna(0)
    merged = merged.dropna(subset=['Customer ID', 'Item', 'Interaction_Score'])
    
    feature_cols = [col for col in merged.columns if col not in ['Customer ID', 'Item', 'Interaction_Score']]
    X = merged[feature_cols]
    y = merged['Interaction_Score']
    
    print(f"\nCreated dataset shape: {X.shape}, NaN check: {X.isna().sum().sum()}")
    print("=> Finished create_ml_dataset")
    return X, y, merged[['Customer ID', 'Item']]

def train_ml_models():
    """Define and initialize ML models with Stacking Ensemble."""
    base_models = [
        ('LightGBM', LGBMRegressor(random_state=42, verbose=-1)),
        ('XGBoost', XGBRegressor(random_state=42, verbosity=0)),
        ('CatBoost', CatBoostRegressor(verbose=0, random_state=42))
    ]
    return {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42, verbose=-1),
        'XGBoost': XGBRegressor(random_state=42, verbosity=0),
        'CatBoost': CatBoostRegressor(verbose=0, random_state=42),
        'Stacking Ensemble': StackingRegressor(estimators=base_models, final_estimator=Ridge(), cv=3)
    }
    
def analyze_ml_by_loyalty(test_df, X_test, y_test, models, user_profiles):
    results = {}
    loyalty_groups = pd.qcut(user_profiles['Customer_Loyalty_Score'], q=3, labels=['Low', 'Medium', 'High'])
    test_df['Loyalty_Group'] = loyalty_groups.loc[test_df['Customer_ID']].values
    
    for group in ['Low', 'Medium', 'High']:
        group_idx = test_df['Loyalty_Group'] == group
        X_group = X_test[group_idx]
        y_group = y_test[group_idx]
        for model_name, model in models.items():
            y_pred = model.predict(X_group)
            rmse = np.sqrt(mean_squared_error(y_group, y_pred))
            results[f"{model_name}_{group}"] = rmse
    return results

def measure_performance(func, *args):
    """Measure performance of a function."""
    start_time = time.time()
    process = psutil.Process()
    cpu_percentages = [psutil.cpu_percent(interval=None)]
    ram_usages = [process.memory_info().rss / 1024 / 1024]
    result = func(*args)
    cpu_percentages.append(psutil.cpu_percent(interval=None))
    ram_usages.append(process.memory_info().rss / 1024 / 1024)
    duration = time.time() - start_time
    cpu_avg = np.mean(cpu_percentages)
    ram_used = max(ram_usages) - min(ram_usages)
    ram_avg = np.mean(ram_usages)
    return result, duration, cpu_avg, ram_used, ram_avg

def get_model_size(model, filename='temp_model.joblib'):
    """Calculate model size."""
    try:
        joblib.dump(model, filename)
        size_mb = os.path.getsize(filename) / 1024 / 1024
        os.remove(filename)
        return size_mb
    except Exception:
        return 0

def recommend_ml_top_n(model, user_id, items, user_profiles, item_features, N=5):
    """Get top-N recommendations for ML models."""
    if user_id not in user_profiles.index:
        user_feat = user_profiles.mean()
    else:
        user_feat = user_profiles.loc[user_id]
    
    input_rows = []
    for item in items:
        item_feat = item_features.loc[item]
        row = pd.concat([user_feat.rename(lambda x: f"{x}_user"), item_feat.rename(lambda x: f"{x}_item")])
        input_rows.append(row)
    input_df = pd.DataFrame(input_rows)
    preds = model.predict(input_df)
    item_scores = list(zip(items, preds))
    top_n = sorted(item_scores, key=lambda x: x[1], reverse=True)[:N]
    return [item for item, score in top_n], [score for item, score in top_n]

def evaluate_ranking_vectorized(model, test_df, items, category_cols, X_test=None, y_test=None, N=5):
    """Evaluate ranking metrics for models, using  for CF and Hybrid models."""
    is_ml = hasattr(model, 'fit') and hasattr(model, 'predict') and model.__class__.__name__ not in ['CBModel', 'ItemCFModel', 'HybridModel']
    results = []
    for user_id in test_df['Customer ID'].unique():
        user_data = test_df[test_df['Customer ID'] == user_id]
        if 'Item' in user_data.columns:
            gt_items = user_data['Item'].dropna().apply(lambda x: x if not isinstance(x, list) else x[0] if x else None).dropna().unique()
            gt_items = [item for item in gt_items if item in items]
        else:
            gt_items = []
        if not gt_items:
            continue
        if is_ml:
            rec_items, rec_scores = recommend_ml_top_n(model, user_id, items, user_profiles, item_features, N)
        else:
            scores = []
            for item in items:
                score = model.predict(user_id, [item])[0]
                scores.append((item, score))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[:N]
            rec_items = [s[0] for s in scores]
        hits = len(set(rec_items) & set(gt_items))
        precision = hits / N
        recall = hits / len(gt_items) if gt_items else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        relevance = [1 if r in gt_items else 0 for r in rec_items]
        ideal = sorted(relevance, reverse=True)
        dcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance)])
        idcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal)])
        ndcg = dcg / idcg if idcg > 0 else 0
        mrr = 0
        for i, rec in enumerate(rec_items, 1):
            if rec in gt_items:
                mrr = 1 / i
                break
        hit = 1 if hits > 0 else 0
        results.append({
            'Precision@N': precision,
            'Recall@N': recall,
            'F1@N': f1,
            'NDCG@N': ndcg,
            'MRR': mrr,
            'Hit Rate': hit
        })
    metrics = pd.DataFrame(results).mean().to_dict() if results else {}
    if is_ml and y_test is not None:
        preds = model.predict(X_test)
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, preds))
        metrics['MAE'] = mean_absolute_error(y_test, preds)
        metrics['R² Score'] = r2_score(y_test, preds)
        n = X_test.shape[0]
        p = X_test.shape[1]
        metrics['Adjusted R²'] = 1 - (1 - metrics['R² Score']) * (n - 1) / (n - p - 1)
        metrics['MAPE (%)'] = np.mean(np.abs((y_test - preds) / y_test)) * 100 if (y_test != 0).all() else np.nan
        metrics['Explained Variance'] = explained_variance_score(y_test, preds)
    return metrics

def evaluate_coverage(model, test_df, items, N=5, user_profiles=None, item_features=None, model_type="cb", ml_model=None, ):
    """Evaluate coverage metric, using  for CF and Hybrid models."""
    recommended_items = set()
    for user_id in test_df['Customer ID'].unique():
        if model_type == "ml":
            rec_items, _ = recommend_ml_top_n(ml_model, user_id, items, user_profiles, item_features, N)
            recommended_items.update(rec_items)
            continue
        scores = []
        for item in items:
            if model_type == "cb":
                score = model.predict(user_id, [item])[0]
            elif model_type == "itemcf":
                score = model.predict(user_id, [item])[0]
            else:
                score = model.predict(user_id, [item])[0]
            if not np.isnan(score) and score > 0:
                scores.append((item, score))
        if scores:
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[:N]
            recommended_items.update([s[0] for s in scores])
    return len(recommended_items) / len(items) if items else 0

def train_model(model, X_train, y_train):
    """Train a model with performance measurement."""
    def _train():
        if hasattr(model, 'fit'):
            return model.fit(X_train, y_train)
        return model
    return measure_performance(_train)

def predict_model(trained_model, X_test, test_df, items):
    """Predict with a model."""
    def _predict():
        is_ml = (
            hasattr(trained_model, 'fit')
            and hasattr(trained_model, 'predict')
            and trained_model.__class__.__name__ not in ['CBModel', 'ItemCFModel', 'HybridModel']
        )
        if is_ml:
            return trained_model.predict(X_test)
        else:
            preds = []
            for idx, row in test_df.iterrows():
                user_id = row['Customer ID']
                item_id = row['Item'] if 'Item' in row else items[0]
                preds.append(trained_model.predict(user_id, [item_id])[0])
            return np.array(preds)
    return measure_performance(_predict)

def get_top_5_with_scores(model, user_id, items, user_profiles, item_features, N=5):
    """Get top-5 recommendations with scores."""
    if model.__class__.__name__ in ['CBModel', 'ItemCFModel', 'HybridModel']:
        scores = model.predict(user_id, items)
        item_scores = list(zip(items, scores))
        top_5 = sorted(item_scores, key=lambda x: x[1], reverse=True)[:N]
        return [(item, score) for item, score in top_5]
    else:
        rec_items, rec_scores = recommend_ml_top_n(model, user_id, items, user_profiles, item_features, N)
        return list(zip(rec_items, rec_scores))

def evaluate_cb_cf_hybrid(test_df, user_profiles, item_features, user_item_matrix, all_items, category_cols, season_cols, age_col, previous_purchases_col, loyalty_score_col, train_df):
    """Evaluate CB, CF, and Hybrid models, using ."""
    cb_model = CBModel(user_profiles, item_features, category_cols, season_cols, age_col, previous_purchases_col, loyalty_score_col, train_df)
    item_cf_model = ItemCFModel(user_item_matrix)
    hybrid_model = HybridModel(user_profiles, item_features, user_item_matrix, alpha=0.7, beta=0.3, category_cols=category_cols, season_cols=season_cols, age_col=age_col, previous_purchases_col=previous_purchases_col, loyalty_score_col=loyalty_score_col, train_df=train_df)
    
    models = [
        ('CB Model', cb_model),
        ('Item-CF Model', item_cf_model),
        ('Hybrid Model', hybrid_model)
    ]
    
    cb_cf_hybrid_results = []
    for name, model in models:
        metrics = evaluate_ranking_vectorized(model, test_df, all_items, category_cols, )
        coverage = evaluate_coverage(model, test_df, all_items, N=5, user_profiles=user_profiles, item_features=item_features, model_type="cb" if name == "CB Model" else "itemcf" if name == "Item-CF Model" else "hybrid")
        filtered_metrics = {k: v for k, v in metrics.items() if k in ['Precision@N', 'Recall@N', 'F1@N', 'NDCG@N', 'MRR', 'Hit Rate']}
        row = {'Model': name, **filtered_metrics, 'Coverage': coverage}
        cb_cf_hybrid_results.append(row)
    
    print("=> Finished evaluate_cb_cf_hybrid_model")
    return pd.DataFrame(cb_cf_hybrid_results), models

def evaluate_ml_models(X_train, y_train, X_test, y_test, test_df, user_profiles, item_features, all_items):
    """Evaluate ML models."""
    ml_models = train_ml_models()
    results = []
    cpu_cores = psutil.cpu_count()
    
    missing_users = set(test_df['Customer ID'].unique()) - set(user_profiles.index)
    if missing_users:
        print(f"\nWarning: user_ids don't have in user_profiles: {missing_users}")
    
    for name, model in ml_models.items():
        trained_model, train_time, train_cpu_avg, train_ram_used, ram_avg = train_model(model, X_train, y_train)
        model_size = get_model_size(trained_model, f'temp_{name.replace(" ", "_")}.joblib')
        
        def _predict():
            return trained_model.predict(X_test)
        preds, inference_time, inference_cpu_avg, _, inference_ram_avg = measure_performance(_predict)
        
        ranking_metrics = evaluate_ranking_vectorized(trained_model, test_df, all_items, category_cols, X_test, y_test)
        coverage = evaluate_coverage(trained_model, test_df, all_items, N=5, user_profiles=user_profiles, item_features=item_features, model_type="ml", ml_model=trained_model)
        results.append({
            'Model': name,
            **ranking_metrics,
            'Coverage': coverage,
            'Train Time (s)': train_time,
            'Model Size (MB)': model_size,
            'CPU Cores': cpu_cores,
            'Train CPU Avg (%)': train_cpu_avg,
            'Train RAM Used (MB)': train_ram_used,
            'Train RAM Avg (%)': ram_avg / (psutil.virtual_memory().total / 1024 / 1024) * 100,
            'Inference Time (s)': inference_time,
            'Inference CPU Avg (%)': inference_cpu_avg,
            'Inference RAM Avg (%)': inference_ram_avg / (psutil.virtual_memory().total / 1024 / 1024) * 100,
        })
    print("=> Finished evaluate_ml_models")
    return pd.DataFrame(results)

def visualize_recommendation_metrics(cb_cf_hybrid_results, results_df):
    """Visualize recommendation metrics."""
    metrics = ['Precision@N', 'Recall@N', 'F1@N', 'NDCG@N', 'MRR', 'Hit Rate']
    plot_data = pd.concat([cb_cf_hybrid_results[['Model'] + metrics], results_df[['Model'] + metrics]], ignore_index=True)
    
    colors = {
        'Precision@N': '#1f77b4',
        'Recall@N': '#ff7f0e',
        'F1@N': '#2ca02c',
        'NDCG@N': '#d62728',
        'MRR': '#9467bd',
        'Hit Rate': '#8c564b'
    }
    
    labels_vn = {
        'Precision@N': 'Precision@N',
        'Recall@N': 'Recall@N',
        'F1@N': 'F1@N',
        'NDCG@N': 'NDCG@N',
        'MRR': 'MRR',
        'Hit Rate': 'Hit Rate'
    }
    
    plt.figure(figsize=(14, 8))
    n_models = len(plot_data)
    n_metrics = len(metrics)
    bar_width = 0.12
    index = np.arange(n_models)
    
    for i, metric in enumerate(metrics):
        bars = plt.bar(index + i * bar_width, plot_data[metric], bar_width,
                       label=labels_vn[metric], color=colors[metric])
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.title('Recommendation Metrics', fontsize=14)
    plt.xticks(index + bar_width * (n_metrics - 1) / 2, plot_data['Model'], rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    plt.tight_layout()
    save_path = os.path.join(TRAINING_DIR, 'recommendation_metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("=> Finished visualize_recommendation_metrics")

def demo_recommendations(demo_users, test_df, user_profiles, all_items, models, tuned_models=None):
    """Demo recommendations for sample users."""
    available_users = [uid for uid in demo_users if uid in test_df['Customer ID'].values]
    if len(available_users) < len(demo_users):
        print(f"Warning: Some users are not in test_df: {set(demo_users) - set(available_users)}")
    
    title = "\n--- DEMO result recommendation of (CB/CF/Hybrid + ML/ensemble) on test set (Before Tuning) ---" if tuned_models is None else "\n--- DEMO result recommendation of (CB/CF/Hybrid + ML/ensemble) on test set (After Tuning) ---"
    print(title)
    for user_id in available_users:
        print(f"\nUser {user_id}:")
        gt_items = test_df[test_df['Customer ID'] == user_id]['Item'].iloc[0] if user_id in test_df['Customer ID'].values else []
        gt_items = gt_items if isinstance(gt_items, list) else [gt_items] if pd.notna(gt_items) else []
        print(f"  Bought: {gt_items}")
        models_to_demo = models if tuned_models is None else models + [(name, model) for name, model in tuned_models.items()]
        for name, model in models_to_demo:
            top_5 = get_top_5_with_scores(model, user_id, all_items, user_profiles, item_features, N=5)
            print(f"  {name} Recommend Top-5:")
            for rank, (item, score) in enumerate(top_5, 1):
                print(f"    {rank}. {item:<25} | Score: {score:.4f}")
    print("=> Finished demo_recommendations")

def kfold_cross_validation(X_train, y_train, test_df, all_items, category_cols):
    """Perform K-Fold Cross-Validation for ML models."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    ml_models = train_ml_models()
    
    for name, model in ml_models.items():
        precision_scores, recall_scores, f1_scores, ndcg_scores = [], [], [], []
        mrr_scores, hit_rate_scores = [], []
        rmse_scores, mae_scores, r2_scores, adjusted_r2_scores, mape_scores, explained_variance_scores = [], [], [], [], [], []
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            val_df = test_df.iloc[val_idx] if len(test_df) == len(X_train) else test_df.iloc[np.random.choice(len(test_df), len(val_idx))]
            
            model.fit(X_tr, y_tr)
            metrics = evaluate_ranking_vectorized(model, val_df, all_items, category_cols, X_val, y_val)
            
            precision_scores.append(metrics.get('Precision@N', 0))
            recall_scores.append(metrics.get('Recall@N', 0))
            f1_scores.append(metrics.get('F1@N', 0))
            ndcg_scores.append(metrics.get('NDCG@N', 0))
            mrr_scores.append(metrics.get('MRR', 0))
            hit_rate_scores.append(metrics.get('Hit Rate', 0))
            rmse_scores.append(metrics.get('RMSE', np.nan))
            mae_scores.append(metrics.get('MAE', np.nan))
            r2_scores.append(metrics.get('R² Score', np.nan))
            adjusted_r2_scores.append(metrics.get('Adjusted R²', np.nan))
            mape_scores.append(metrics.get('MAPE (%)', np.nan))
            explained_variance_scores.append(metrics.get('Explained Variance', np.nan))
        
        cv_results.append({
            'Model': name,
            'Precision@N Mean': np.mean(precision_scores),
            'Precision@N Std': np.std(precision_scores),
            'Recall@N Mean': np.mean(recall_scores),
            'Recall@N Std': np.std(recall_scores),
            'F1@N Mean': np.mean(f1_scores),
            'F1@N Std': np.std(f1_scores),
            'NDCG@N Mean': np.mean(ndcg_scores),
            'NDCG@N Std': np.std(ndcg_scores),
            'MRR Mean': np.mean(mrr_scores),
            'MRR Std': np.std(mrr_scores),
            'Hit Rate Mean': np.mean(hit_rate_scores),
            'Hit Rate Std': np.std(hit_rate_scores),
            'RMSE Mean': np.nanmean(rmse_scores),
            'RMSE Std': np.nanstd(rmse_scores),
            'MAE Mean': np.nanmean(mae_scores),
            'MAE Std': np.nanstd(mae_scores),
            'R² Score Mean': np.nanmean(r2_scores),
            'R² Score Std': np.nanstd(r2_scores),
            'Adjusted R² Mean': np.nanmean(adjusted_r2_scores),
            'Adjusted R² Std': np.nanstd(adjusted_r2_scores),
            'MAPE (%) Mean': np.nanmean(mape_scores),
            'MAPE (%) Std': np.nanstd(mape_scores),
            'Explained Variance Mean': np.nanmean(explained_variance_scores),
            'Explained Variance Std': np.nanstd(explained_variance_scores)
        })
    
    return pd.DataFrame(cv_results)

def visualize_kfold_results(cv_results_df):
    """Visualize K-Fold Cross-Validation results."""
    metric_labels_kfold = {
        'Precision@N Mean': 'Precision@N',
        'Recall@N Mean': 'Recall@N',
        'F1@N Mean': 'F1@N',
        'NDCG@N Mean': 'NDCG@N',
        'MRR Mean': 'MRR',
        'Hit Rate Mean': 'Hit Rate'
    }
    
    metrics = [m for m in metric_labels_kfold.keys() if m in cv_results_df.columns]
    plot_data = cv_results_df.set_index('Model')
    
    new_labels = [metric_labels_kfold.get(m, m) for m in metrics]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    bars = plot_data[metrics].plot(
        kind='barh',
        yerr=plot_data[[m.replace('Mean', 'Std') for m in metrics]],
        ax=ax,
        width=0.7,
        legend=False
    )
    ax.set_title('Recommendation Metrics (K-Fold CV)', fontsize=18)
    ax.set_xlabel('Values', fontsize=14)
    ax.set_ylabel('Models', fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    handles, labels_ = bars.get_legend_handles_labels()
    ax.legend(
        handles, new_labels,
        title='Metrics',
        bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, title_fontsize=13, frameon=True
    )
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    save_path = os.path.join(TRAINING_DIR, 'K_Fold_result.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def tune_hybrid_model(user_profiles, item_features, user_item_matrix, test_df, all_items, category_cols, season_cols, age_col, previous_purchases_col, loyalty_score_col, train_df):
    """Tune alpha and beta for Hybrid Model using Optuna."""
    def objective(trial):
        alpha = trial.suggest_float('alpha', 0.1, 0.9)
        beta = 1 - alpha
        hybrid_model = HybridModel(
            user_profiles, item_features, user_item_matrix,
            alpha=alpha, beta=beta, category_cols=category_cols, season_cols=season_cols,
            age_col=age_col, previous_purchases_col=previous_purchases_col, loyalty_score_col=loyalty_score_col, train_df=train_df
        )
        metrics = evaluate_ranking_vectorized(hybrid_model, test_df, all_items, category_cols, )
        return metrics['NDCG@N']
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    best_alpha = best_params['alpha']
    best_beta = 1 - best_alpha
    tuned_hybrid_model = HybridModel(
        user_profiles, item_features, user_item_matrix,
        best_alpha, best_beta, category_cols, season_cols, age_col, previous_purchases_col, loyalty_score_col, train_df
    )
    print(f"Tuned Hybrid Model: alpha={best_alpha:.3f}, beta={best_beta:.3f}")
    return tuned_hybrid_model, best_params

def tune_ml_models(X_train, y_train, test_df, all_items, category_cols, user_profiles, item_features):
    """Fine-tune ML models using Optuna."""
    param_grids = {
        'Ridge Regression': {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'LightGBM': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 7, -1],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0]
        },
        'CatBoost': {
            'iterations': [50, 100, 200],
            'depth': [3, 6, 10],
            'learning_rate': [0.01, 0.05, 0.1]
        },
        'Stacking Ensemble': {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        }
    }
    
    def optuna_objective_factory(name, model, X, y, param_grid):
        def objective(trial):
            params = {}
            for param, values in param_grid.items():
                params[param] = trial.suggest_categorical(param, values)
            if name == "CatBoost":
                _model = CatBoostRegressor(verbose=0, random_state=42, **params)
            elif name == "LightGBM":
                _model = LGBMRegressor(random_state=42, verbose=-1, **params)
            elif name == "XGBoost":
                _model = XGBRegressor(random_state=42, verbosity=0, **params)
            elif name == "Random Forest":
                _model = RandomForestRegressor(random_state=42, **params)
            elif name == "Ridge Regression":
                _model = Ridge(**params)
            elif name == "Stacking Ensemble":
                base_models = [
                    ('LightGBM', LGBMRegressor(random_state=42, verbose=-1)),
                    ('XGBoost', XGBRegressor(random_state=42, verbosity=0)),
                    ('CatBoost', CatBoostRegressor(verbose=0, random_state=42))
                ]
                ridge_params = {k: v for k, v in params.items() if k == 'alpha'}
                _model = StackingRegressor(estimators=base_models, final_estimator=Ridge(**ridge_params), cv=3)
            else:
                _model = model.__class__(**params)
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(
                _model, X, y, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
            )
            return scores.mean()
        return objective
    
    tuned_models = {}
    tuned_results = []
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    for name, model in train_ml_models().items():
        print(f"\n--- Tuning model: {name} (Optuna) ---")
        param_grid = param_grids.get(name, {})
        if param_grid:
            study = optuna.create_study(direction='maximize')
            study.optimize(
                optuna_objective_factory(name, model, X_train, y_train, param_grid),
                n_trials=30,
                show_progress_bar=True
            )
            best_params = study.best_params
            if name == "CatBoost":
                tuned_model = CatBoostRegressor(verbose=0, random_state=42, **best_params)
            elif name == "LightGBM":
                tuned_model = LGBMRegressor(random_state=42, verbose=-1, **best_params)
            elif name == "XGBoost":
                tuned_model = XGBRegressor(random_state=42, verbosity=0, **best_params)
            elif name == "Random Forest":
                tuned_model = RandomForestRegressor(random_state=42, **best_params)
            elif name == "Ridge Regression":
                tuned_model = Ridge(**best_params)
            elif name == "Stacking Ensemble":
                base_models = [
                    ('LightGBM', LGBMRegressor(random_state=42, verbose=-1)),
                    ('XGBoost', XGBRegressor(random_state=42, verbosity=0)),
                    ('CatBoost', CatBoostRegressor(verbose=0, random_state=42))
                ]
                ridge_params = {k: v for k, v in best_params.items() if k == 'alpha'}
                tuned_model = StackingRegressor(estimators=base_models, final_estimator=Ridge(**ridge_params), cv=3)
            else:
                tuned_model = model.__class__(**best_params)
            tuned_model.fit(X_train, y_train)
        else:
            tuned_model = model
            tuned_model.fit(X_train, y_train)
            best_params = {}
        
        # Save tuned model
        joblib.dump(tuned_model, os.path.join(MODELS_DIR, f"tuned_{name.replace(' ', '_')}.pkl"))
        print(f"Saved tuned {name} to {MODELS_DIR}/tuned_{name.replace(' ', '_')}.pkl")
        
        tuned_models[f'Tuned {name}'] = tuned_model
        ranking_metrics = evaluate_ranking_vectorized(
            tuned_model, test_df, all_items, category_cols, X_test, y_test
        )
        coverage = evaluate_coverage(
            tuned_model, test_df, all_items, N=5, user_profiles=user_profiles, item_features=item_features, model_type="ml", ml_model=tuned_model
        )
        tuned_results.append({
            'Model': f'Tuned {name}',
            **ranking_metrics,
            'Coverage': coverage,
            'Best Params': best_params
        })
    
    return pd.DataFrame(tuned_results), tuned_models

def visualize_tuned_results(tuned_comparison_df):
    """Visualize results after tuning."""
    metrics = ['Precision@N', 'Recall@N', 'F1@N', 'NDCG@N', 'MRR', 'Hit Rate']
    metrics = [m for m in metrics if m in tuned_comparison_df.columns]
    plot_data = tuned_comparison_df[['Model'] + metrics].copy()
    n_models = len(plot_data)
    n_metrics = len(metrics)
    
    colors = {
        'Precision@N': '#1f77b4',
        'Recall@N': '#ff7f0e',
        'F1@N': '#2ca02c',
        'NDCG@N': '#d62728',
        'MRR': '#9467bd',
        'Hit Rate': '#8c564b',
    }
    
    bar_width = 0.12
    index = np.arange(n_models)
    
    plt.figure(figsize=(14, 8))
    for i, metric in enumerate(metrics):
        values = plot_data[metric]
        bars = plt.bar(
            index + i * bar_width,
            values,
            bar_width,
            label=metric,
            color=colors.get(metric, '#333333')
        )
        for bar in bars:
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f'{h:.3f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Comparison of Recommendation Metrics', fontsize=14)
    plt.xticks(
        index + bar_width * (n_metrics - 1) / 2,
        plot_data['Model'],
        rotation=45,
        ha='right'
    )
    plt.ylim(0, 1.1)
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_path = os.path.join(TRAINING_DIR, 'Metric_models_after_fine-tuning.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("=> Finished visualize_tuned_results")

def save_best_model(tuned_hybrid_model, tuned_models, cb_cf_hybrid_results, tuned_results_df, 
                    user_profiles, item_features, user_item_matrix, category_cols, season_cols, 
                    age_col, previous_purchases_col, loyalty_score_col, train_df):
    """Lưu mô hình tốt nhất dựa trên F1@N để triển khai."""
    all_results = pd.concat([cb_cf_hybrid_results, tuned_results_df], ignore_index=True)
    sorted_results = all_results.sort_values(by='F1@N', ascending=False)
    
    if sorted_results.empty:
        raise ValueError("There is not best model to choose")
    
    best_model_name = sorted_results.iloc[0]['Model']
    print(f"\nBest model was selected: {best_model_name} (F1@N: {sorted_results.iloc[0]['F1@N']:.3f})")
    
    # Chuẩn bị thông tin mô hình để lưu
    if best_model_name.startswith('Tuned') and best_model_name in tuned_models:
        # Mô hình ML
        best_model = tuned_models[best_model_name]
        model_type = 'ml'
        best_model_info = {
            "model": best_model,
            "model_type": model_type,
            "model_name": best_model_name
        }
    else:
        # Mô hình CB, CF, Hybrid
        config = {
            "category_cols": category_cols,
            "season_cols": season_cols,
            "age_col": age_col,
            "previous_purchases_col": previous_purchases_col,
            "loyalty_score_col": loyalty_score_col,
            "train_df": train_df  # Lưu DataFrame hoặc đường dẫn tới file nếu lớn
        }
        if best_model_name == 'CB Model':
            model_type = 'cb'
        elif best_model_name == 'Item-CF Model':
            model_type = 'itemcf'
            config = {}  # ItemCF chỉ cần user_item_matrix, không cần config phức tạp
        elif best_model_name in ['Hybrid Model', 'Tuned Hybrid Model']:
            model_type = 'hybrid'
            config["alpha"] = 0.7 if best_model_name == 'Hybrid Model' else tuned_hybrid_model.alpha
            config["beta"] = 0.3 if best_model_name == 'Hybrid Model' else tuned_hybrid_model.beta
        else:
            raise ValueError(f"Model name not identified: {best_model_name}")
        
        best_model_info = {
            "model_type": model_type,
            "config": config,
            "model_name": best_model_name
        }
    
    # Lưu mô hình
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model_info, os.path.join(MODELS_DIR, "best_model.pkl"))
    print(f"Save best model ({best_model_name}) to {MODELS_DIR}/best_model.pkl")

def main():
    """Main function to run the training pipeline."""
    global train_df, test_df, user_profiles, item_features, category_cols, all_items, X_test, y_test, user_item_matrix
    
    create_training_dir()
    features_df, user_item_matrix, processed_features_df = load_processed_data()
    features_df, user_item_matrix = validate_data(features_df, processed_features_df, user_item_matrix)
    train_df, test_df = split_train_test(features_df)
    
    user_profile_features, category_cols, season_cols, age_col, previous_purchases_col, loyalty_score_col, interaction_score_col = define_user_item_features(features_df)
    validate_item_categories(features_df, user_item_matrix, category_cols)
    
    print("\nAssign for train_df:")
    train_df['Item'] = assign_item_column(train_df, user_item_matrix, features_df)
    print("\nAssign for test_df:")
    test_df['Item'] = assign_item_column(test_df, user_item_matrix, features_df)
    
    print("\nStarting create user_profiles...")
    user_profiles = create_user_profiles(features_df, user_profile_features, 'Customer_Loyalty_Score')
    print("\nStarting create item_features...")
    item_features = create_item_features(train_df, user_item_matrix, user_profile_features)
    
    save_pipeline_data(user_item_matrix, user_profiles, item_features)
    
    train_df = train_df[train_df['Item'].apply(lambda x: not isinstance(x, float) or not np.isnan(x))].copy()
    test_df = test_df[test_df['Item'].apply(lambda x: not isinstance(x, float) or not np.isnan(x))].copy()
    
    user_item_matrix= prepare_user_item_matrix(train_df, test_df, user_item_matrix)
    all_items = user_item_matrix.columns.tolist()
    
    cb_cf_hybrid_results, models = evaluate_cb_cf_hybrid(
        test_df, user_profiles, item_features, user_item_matrix, all_items,
        category_cols, season_cols, age_col, previous_purchases_col, loyalty_score_col, train_df
    )
    print('\n### CB, CF, Hybrid results:')
    print(cb_cf_hybrid_results)
    
    X_train, y_train, train_pairs = create_ml_dataset(train_df, user_profiles, item_features, train_df, is_train=True)
    X_test, y_test, test_pairs = create_ml_dataset(test_df, user_profiles, item_features, test_df, is_train=False)
    print(f"\nX_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_test.shape: {y_test.shape}")
    
    
    results_df = evaluate_ml_models(X_train, y_train, X_test, y_test, test_df, user_profiles, item_features, all_items)
    print('\n\nMachine learning & Ensemble models results:')
    print(results_df)
    
    comparison_df = pd.concat([results_df, cb_cf_hybrid_results], ignore_index=True)
    print('\n\n=== Metrics comparison among CB/CF/Hybrid and machine learning with ensemble models training:')
    print(comparison_df)
    
    visualize_recommendation_metrics(cb_cf_hybrid_results, results_df)
    
    print("\nStarting demo...")
    demo_users = test_df['Customer ID'].sample(n=min(3, len(test_df)), random_state=42).tolist()
    demo_recommendations(demo_users, test_df, user_profiles, all_items, models)
    
    cv_results_df = kfold_cross_validation(X_train, y_train, test_df, all_items, category_cols)
    print("\nK-Fold Cross-Validation results:")
    print(cv_results_df)
    visualize_kfold_results(cv_results_df)
    
    print("\nStarting fine-tuning...")
    tuned_hybrid_model, best_hybrid_params = tune_hybrid_model(
        user_profiles, item_features, user_item_matrix, test_df, all_items,
        category_cols, season_cols, age_col, previous_purchases_col, loyalty_score_col, train_df
    )
    models.append(('Tuned Hybrid Model', tuned_hybrid_model))
    hybrid_metrics = evaluate_ranking_vectorized(tuned_hybrid_model, test_df, all_items, category_cols, )
    hybrid_coverage = evaluate_coverage(tuned_hybrid_model, test_df, all_items, N=5, user_profiles=user_profiles, item_features=item_features, model_type="hybrid")
    new_row = pd.DataFrame([{
        'Model': 'Tuned Hybrid Model',
        **hybrid_metrics,
        'Coverage': hybrid_coverage,
        'Best Params': best_hybrid_params
    }])
    cb_cf_hybrid_results = pd.concat([cb_cf_hybrid_results, new_row], ignore_index=True)
    
    
    print("\nStarting fine-tuning machine learning models...")
    tuned_results_df, tuned_models = tune_ml_models(X_train, y_train, test_df, all_items, category_cols, user_profiles, item_features)
    print("\nTuned ML and Ensemble models results:")
    print(tuned_results_df)
    
    tuned_comparison_df = pd.concat([cb_cf_hybrid_results, tuned_results_df], ignore_index=True)
    print("\n=== Metrics comparison after fine-tuning:")
    print(tuned_comparison_df)
    
    visualize_tuned_results(tuned_comparison_df)
    
    # thêm mới
    loyalty_analysis = analyze_ml_by_loyalty(test_df, X_test, y_test, tuned_models, user_profiles)
    print("Performance by Loyalty Group:", loyalty_analysis)
    
    demo_recommendations(demo_users, test_df, user_profiles, all_items, models, tuned_models)
    
    save_best_model(
        tuned_hybrid_model, tuned_models, cb_cf_hybrid_results, tuned_results_df,
        user_profiles, item_features, user_item_matrix,
        category_cols, season_cols, age_col, previous_purchases_col, loyalty_score_col, train_df
    )

if __name__ == "__main__":
    main()