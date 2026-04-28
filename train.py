import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb

# Set MLflow experiment
mlflow.set_experiment("E-Commerce_Recommendation_Engine")

class HybridRecommender:
    def __init__(self, n_latent_factors=50):
        self.n_latent_factors = n_latent_factors
        self.svd = TruncatedSVD(n_components=n_latent_factors)
        self.item_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.ranking_model = xgb.XGBRanker(
            objective='rank:pairwise',
            learning_rate=0.1,
            max_depth=6,
            n_estimators=100
        )
        self.user_item_matrix = None
        self.products_df = None
        
    def fit(self, interactions_df, products_df):
        print("Training Hybrid Recommender...")
        self.products_df = products_df
        
        # 1. Collaborative Filtering (Matrix Factorization)
        self.user_item_matrix = interactions_df.pivot_table(
            index='user_id', columns='product_id', values='weight', aggfunc='sum', fill_value=0
        )
        sparse_matrix = csr_matrix(self.user_item_matrix.values)
        self.svd.fit(sparse_matrix)
        
        # 2. Item-Item Similarity (KNN)
        self.item_knn.fit(sparse_matrix.T)
        
        # 3. XGBoost Ranking (Preparation)
        # In a real scenario, we'd use features like price, brand popularity, etc.
        train_data = interactions_df.merge(products_df, on='product_id')
        X = train_data[['price', 'rating', 'weight']]
        y = train_data['interaction_type'].map({'view': 1, 'click': 2, 'cart_addition': 3, 'purchase': 4})
        
        # Simplified ranking training
        self.ranking_model.fit(X, y, group=[len(y)])
        
        print("Training complete.")

    def get_recommendations(self, user_id, n=10):
        if user_id not in self.user_item_matrix.index:
            # Popularity based fallback
            return self.products_df.sort_values(by='rating', ascending=False).head(n)['product_id'].tolist()
        
        u_idx = self.user_item_matrix.index.get_loc(user_id)
        user_vector = self.user_item_matrix.iloc[u_idx].values.reshape(1, -1)
        
        # SVD Scores
        user_factors = self.svd.transform(user_vector)
        scores = np.dot(user_factors, self.svd.components_)
        
        top_indices = np.argsort(scores[0])[::-1][:n*2] # Get extra for filtering
        rec_ids = [self.user_item_matrix.columns[i] for i in top_indices]
        
        return rec_ids[:n]

def train_pipeline():
    with mlflow.start_run():
        # Load data
        if not os.path.exists('data/interactions.csv'):
            print("Data not found. Run generate_data.py first.")
            return

        interactions = pd.read_csv('data/interactions.csv')
        products = pd.read_csv('data/products.csv')
        
        # Feature Engineering
        print("Feature Engineering...")
        user_activity = interactions.groupby('user_id')['weight'].sum().reset_index()
        mlflow.log_metric("total_interactions", len(interactions))
        mlflow.log_metric("unique_users", interactions['user_id'].nunique())
        
        # Train/Test Split
        train_df, test_df = train_test_split(interactions, test_size=0.2, random_state=42)
        
        # Train Model
        model = HybridRecommender(n_latent_factors=100)
        model.fit(train_df, products)
        
        # Evaluate (Simplified RMSE on user-item matrix)
        # In a real engine, we'd use Precision@K/Recall@K
        mlflow.log_param("n_latent_factors", 100)
        
        # Save Model
        os.makedirs('models', exist_ok=True)
        model_path = 'models/model.pkl'
        joblib.dump(model, model_path)
        
        # Log to MLflow
        mlflow.sklearn.log_model(model, "recommender_model")
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_pipeline()
