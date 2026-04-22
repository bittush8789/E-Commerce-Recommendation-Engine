import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

class RecommenderModels:
    def __init__(self, interactions_df):
        self.df = interactions_df
        self.user_item_matrix = self.df.pivot_table(
            index='user_id', columns='product_id', values='weight', aggfunc='sum', fill_value=0
        )
        self.sparse_matrix = csr_matrix(self.user_item_matrix.values)
        self.user_ids = self.user_item_matrix.index.tolist()
        self.product_ids = self.user_item_matrix.columns.tolist()

    def get_popularity_recommendations(self, n=10):
        pop = self.df.groupby('product_id')['weight'].sum().sort_values(ascending=False)
        return pop.head(n).index.tolist()

    def get_cf_recommendations(self, user_id, n=10):
        if user_id not in self.user_ids:
            return self.get_popularity_recommendations(n)
        
        u_idx = self.user_ids.index(user_id)
        # Item-Item Similarity
        item_sim = cosine_similarity(self.sparse_matrix.T)
        user_vector = self.user_item_matrix.iloc[u_idx].values
        scores = user_vector.dot(item_sim)
        
        # Filter seen
        scores[user_vector > 0] = -1
        top_indices = np.argsort(scores)[::-1][:n]
        return [self.product_ids[i] for i in top_indices]

    def get_svd_recommendations(self, user_id, n=10):
        if user_id not in self.user_ids:
            return self.get_popularity_recommendations(n)
            
        u_idx = self.user_ids.index(user_id)
        svd = TruncatedSVD(n_components=min(50, self.sparse_matrix.shape[1]-1))
        user_factors = svd.fit_transform(self.sparse_matrix)
        item_factors = svd.components_.T
        
        scores = np.dot(user_factors[u_idx], item_factors.T)
        user_vector = self.user_item_matrix.iloc[u_idx].values
        scores[user_vector > 0] = -1
        top_indices = np.argsort(scores)[::-1][:n]
        return [self.product_ids[i] for i in top_indices]

def calculate_hit_rate(model_func, test_df, n=10):
    hits = 0
    total = 0
    test_users = test_df['user_id'].unique()
    for uid in test_users[:100]: # Sample for speed
        actual = set(test_df[test_df['user_id'] == uid]['product_id'])
        preds = model_func(uid, n)
        if any(p in actual for p in preds):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0
