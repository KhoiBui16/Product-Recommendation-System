# Hybrid Product Recommendation System

A hybrid product recommendation system that analyzes customer behavior and shopping habits to deliver personalized product suggestions for e-commerce. The system combines content-based, collaborative filtering, and machine learning approaches to maximize personalization and recommendation accuracy.

---

## 📝 Description & System Overview

This project leverages the [Consumer Behavior and Shopping Habits Dataset](https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset) (~3900 rows, 18 columns), containing key customer demographics, product interactions, ratings, and purchase histories.

**Key Features:**
- Unified Streamlit web interface: view product list (list/grid), filter, and receive personalized product recommendations.
- Hybrid recommendation engine: content-based, collaborative filtering, ensemble ML (LightGBM, CatBoost, XGBoost), and hybrid weighted scoring.
- Advanced feature engineering and preprocessing for categorical and numerical attributes.
- Model evaluation using ranking metrics (Precision@N, Recall@N, NDCG, Coverage, RMSE, etc.).
- Modular, maintainable code structure for easy extension.

**Video Demo:**  
[👉 Watch the demo video here](https://youtu.be/GDIEp9cTHi4?si=EW8wvRWOflUlzfnG)

**Website Demo:**  
[👉 website demo here](https://personalized-recommendation-system.streamlit.app/)

---

## 📁 Project Directory Structure

```
Product-Recommendation-System/
│
├── app.py                                              
├── requirements.txt                                    
├── README.md                                           
│
├── data/                                               
│   ├── shopping_behavior_updated.csv
│   ├── shopping_behavior_processed.csv
│   ├── shopping_behavior_final_features.csv
│   └── user_item_matrix.csv
│
├── src/                                                
│   ├── __init__.py
│   ├── eda.py                                          
│   ├── preprocessing.py                                
│   ├── feature_engineering.py                          
│   ├── training.py                                     
│   ├── recommendation_models.py                        
│   └── logger.py                                       
│
├── Models/                                             
│   ├── best_model.pkl
│   ├── user_profiles.pkl
│   ├── item_features.pkl
│   └── user_item_matrix.pkl
│   └── train_df.pkl
│
├── logs/                                               
│   └── *.log
│
├── Figures/                                            
│   ├── EDA/
│   ├── Preprocessing/
│   ├── Feature_Engineering/
│   └── Training/
│
└── notebooks/                                          
    └── *.ipynb
```

---

## 🚀 Local Setup & Running Instructions

### 1. Clone the repository

```bash
git clone https://github.com/KhoiBui16/Product-Recommendation-System.git
cd Product-Recommendation-System
```

### 2. (Optional) Set up a Python virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/MacOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

- Make sure the `Data/` folder contains at least `shopping_behavior_updated.csv`.  
- Download from [Kaggle](https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset) if needed.

### 5. Run the application

```bash
streamlit run app.py
```

- The app will be available at: http://localhost:8501

---

## 🤝 Contact & Contributions

- **Main Author:** [KhoiBui16] ([khoib1601@gmail.com](mailto:khoib1601@gmail.com))
- **Contributions, Issues, Ideas:**  
  Please open an issue or pull request on this GitHub repository, or contact via email above.
