import logging
import pandas as pd
import os
import joblib
import numpy as np
from .logger import setup_logger
from .training import MODELS_DIR, CBModel, ItemCFModel, HybridModel

# Thiết lập logger
logger = setup_logger(__name__, 'recommendation_models.log')

# Paths to pickle files
BEST_MODEL_FILE = os.path.join(MODELS_DIR, "best_model.pkl")
USER_PROFILES_FILE = os.path.join(MODELS_DIR, "user_profiles.pkl")
ITEM_FEATURES_FILE = os.path.join(MODELS_DIR, "item_features.pkl")
USER_ITEM_MATRIX_FILE = os.path.join(MODELS_DIR, "user_item_matrix.pkl")
TRAIN_DF_FILE = os.path.join(MODELS_DIR, "train_df.pkl")

def reverse_encode_category(df):
    """Reverse encode category columns to a single 'Category' column."""
    logger.info("Reverse encoding category columns...")
    try:
        category_columns = [col for col in df.columns if col.startswith("Category_")]
        if category_columns:
            df["Category"] = df[category_columns].idxmax(axis=1).str.replace("Category_", "")
            logger.info("Reverse encoding category completed")
        else:
            logger.warning("No Category_ columns found in dataframe")
        return df
    except Exception as e:
        logger.error(f"Error in reverse_encode_category: {str(e)}")
        raise

def reverse_encode_review_rating(df):
    """Convert Review Rating column to float."""
    logger.info("Converting Review Rating to float...")
    try:
        if "Review Rating" in df.columns:
            df["Review Rating"] = df["Review Rating"].astype(float)
            logger.info("Review Rating conversion completed")
        else:
            logger.warning("Review Rating column not found in dataframe")
        return df
    except Exception as e:
        logger.error(f"Error in reverse_encode_review_rating: {str(e)}")
        raise

def load_pickle(file_path):
    """Load a pickle file with error handling and logging."""
    logger.info(f"Đang nạp file pickle từ {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Missing file: {file_path}")
    try:
        data = joblib.load(file_path)
        logger.info(f"Nạp file {file_path} thành công")
        return data
    except Exception as e:
        logger.error(f"Lỗi khi nạp file {file_path}: {str(e)}")
        raise

def save_pickle(data, file_path):
    """Save a pickle file with error handling and logging."""
    logger.info(f"Đang lưu file pickle vào {file_path}")
    try:
        joblib.dump(data, file_path)
        logger.info(f"Lưu file {file_path} thành công")
    except Exception as e:
        logger.error(f"Lỗi khi lưu file {file_path}: {str(e)}")
        raise

class RecommendationModel:
    def __init__(self):
        """Load best_model.pkl, pipeline data, and instantiate the correct model."""
        logger.info("Khởi tạo RecommendationModel...")
        try:
            # Load model_info
            self.model_info = load_pickle(BEST_MODEL_FILE)
            self.model_type = self.model_info.get("model_type")
            self.model_name = self.model_info.get("model_name", "Unknown Model")
            logger.info(f"Model info loaded: {self.model_name} (type: {self.model_type})")

            # Load pipeline data
            self.user_profiles = load_pickle(USER_PROFILES_FILE)
            self.item_features = load_pickle(ITEM_FEATURES_FILE)
            self.user_item_matrix = load_pickle(USER_ITEM_MATRIX_FILE)
            self.all_items = self.user_item_matrix.columns.tolist()
            logger.info("Pipeline data loaded successfully")

            # Instantiate model
            cfg = self.model_info.get('config', {})
            train_df_path = TRAIN_DF_FILE
            if 'train_df' in cfg and not os.path.exists(train_df_path):
                logger.info(f"Saving train_df from config to {train_df_path}")
                save_pickle(cfg['train_df'], train_df_path)
            
            if self.model_type == 'ml':
                self.model = self.model_info['model']
                logger.info("ML model instantiated")
            elif self.model_type == 'cb':
                self.model = CBModel(
                    self.user_profiles,
                    self.item_features,
                    cfg.get('category_cols', []),
                    cfg.get('season_cols', []),
                    cfg.get('age_col', ''),
                    cfg.get('previous_purchases_col', ''),
                    cfg.get('loyalty_score_col', ''),
                    train_df_path
                )
                logger.info("Content-Based model instantiated")
            elif self.model_type == 'itemcf':
                self.model = ItemCFModel(self.user_item_matrix)
                logger.info("Item-CF model instantiated")
            elif self.model_type == 'hybrid':
                self.model = HybridModel(
                    self.user_profiles,
                    self.item_features,
                    self.user_item_matrix,
                    alpha=cfg.get('alpha', 0.5),
                    beta=cfg.get('beta', 0.5),
                    category_cols=cfg.get('category_cols', []),
                    season_cols=cfg.get('season_cols', []),
                    age_col=cfg.get('age_col', ''),
                    previous_purchases_col=cfg.get('previous_purchases_col', ''),
                    loyalty_score_col=cfg.get('loyalty_score_col', ''),
                    train_df=train_df_path
                )
                logger.info("Hybrid model instantiated")
            else:
                logger.error(f"Unsupported model_type: {self.model_type}")
                raise ValueError(f"Unsupported model_type: {self.model_type}")

            logger.info(f"RecommendationModel initialized: {self.model_name} (type: {self.model_type})")
        except Exception as e:
            logger.error(f"Error initializing RecommendationModel: {str(e)}")
            raise

    def get_recommendations(self, customer_id, top_n=5):
        """Return top-n recommendations for a given customer_id."""
        logger.info(f"Generating recommendations for customer_id: {customer_id}, top_n: {top_n}")
        try:
            # Handle unknown user
            if customer_id not in self.user_profiles.index:
                logger.warning(f"Unknown customer_id: {customer_id}, using default scores")
                default_score = self.user_item_matrix.values.mean()
                recs = [(itm, default_score) for itm in self.all_items]
            else:
                if self.model_type == 'ml':
                    from .training import recommend_ml_top_n
                    logger.info("Using ML model for recommendations")
                    recs, _ = recommend_ml_top_n(
                        self.model,
                        customer_id,
                        self.all_items,
                        self.user_profiles,
                        self.item_features,
                        N=len(self.all_items)
                    )
                else:
                    logger.info(f"Using {self.model_type} model for predictions")
                    scores = self.model.predict(customer_id, self.all_items)
                    recs = list(zip(self.all_items, scores))
            # Sort and return top_n
            recs = sorted(recs, key=lambda x: x[1], reverse=True)[:top_n]
            logger.info(f"Top {top_n} recommendations generated: {recs}")
            return recs
        except Exception as e:
            logger.error(f"Error in get_recommendations: {str(e)}")
            raise

    def get_content_based_recommendations(self, filters, top_n=5):
        """Apply content filters on raw CSV and return top-n items by Review Rating."""
        logger.info(f"Generating content-based recommendations with filters: {filters}, top_n: {top_n}")
        try:
            # Load raw data to get original Review Rating (0-5 scale)
            raw_csv = os.path.join(os.path.dirname(__file__), '..', 'Data', 'shopping_behavior_updated.csv')
            if not os.path.exists(raw_csv):
                logger.error(f"Raw CSV not found: {raw_csv}")
                raise FileNotFoundError(f"Raw CSV not found: {raw_csv}")
            df_raw = pd.read_csv(raw_csv)
            logger.info(f"Loaded raw CSV: {raw_csv}")

            # Apply filters
            df_filtered = df_raw.copy()
            for col, val in filters.items():
                logger.info(f"Applying filter: {col} = {val}")
                if col == 'Review Rating':
                    df_filtered = df_filtered[df_filtered[col] >= val]
                else:
                    df_filtered = df_filtered[df_filtered[col] == val]
            
            if df_filtered.empty:
                logger.warning("No items match the filters")
                return []
            
            # Group by Item Purchased to get average Review Rating
            item_scores = df_filtered.groupby('Item Purchased')['Review Rating'].mean().reset_index()
            # Sort by Review Rating and get top_n
            item_scores = item_scores.sort_values(by='Review Rating', ascending=False)
            recs = list(zip(item_scores['Item Purchased'], item_scores['Review Rating']))[:top_n]
            logger.info(f"Content-based recommendations: {recs}")
            return recs
        except Exception as e:
            logger.error(f"Error in get_content_based_recommendations: {str(e)}")
            raise