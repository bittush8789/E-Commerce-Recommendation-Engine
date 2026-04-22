import pandas as pd
import numpy as np
import pickle
import os
import random
from datetime import datetime, timedelta
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def generate_data():
    print("--- Step 1: Generating High-Quality Synthetic Data ---")
    num_users = 3000
    num_products = 800
    
    users = pd.DataFrame({
        'user_id': range(1, num_users + 1),
        'preferred_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], size=num_users)
    })
    
    products = pd.DataFrame({
        'product_id': range(1, num_products + 1),
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], size=num_products),
        'base_popularity': np.random.exponential(scale=1.0, size=num_products)
    })
    
    # Generate Interactions with specific patterns
    events_list = []
    for _, user in users.iterrows():
        # Users interact more with their preferred category
        pref_cat = user['preferred_category']
        cat_products = products[products['category'] == pref_cat]['product_id'].values
        other_products = products[products['category'] != pref_cat]['product_id'].values
        
        # Number of interactions per user
        n_inter = np.random.randint(5, 30)
        for _ in range(n_inter):
            if random.random() < 0.7: # 70% chance to pick from preferred category
                pid = np.random.choice(cat_products)
            else:
                pid = np.random.choice(other_products)
                
            etype = np.random.choice(['view', 'click', 'add_to_cart', 'purchase'], p=[0.5, 0.3, 0.15, 0.05])
            events_list.append([user['user_id'], pid, etype, datetime.now()])
            
    events = pd.DataFrame(events_list, columns=['user_id', 'product_id', 'event_type', 'timestamp'])
    event_weights = {'view': 1, 'click': 2, 'add_to_cart': 5, 'purchase': 10}
    events['weight'] = events['event_type'].map(event_weights)
    
    users.to_csv(os.path.join(DATA_DIR, 'users_v2.csv'), index=False)
    products.to_csv(os.path.join(DATA_DIR, 'products_v2.csv'), index=False)
    events.to_csv(os.path.join(DATA_DIR, 'interactions_v2.csv'), index=False)
    return users, products, events

def train_and_evaluate(users, products, events):
    print("\n--- Step 2: Training & Evaluating Multiple Models ---")
    
    # Train-Test Split by time or random
    train_events, test_events = train_test_split(events, test_size=0.2, random_state=42)
    
    # Global Popularity
    pop_df = train_events.groupby('product_id')['weight'].sum().reset_index().sort_values('weight', ascending=False)
    top_10_pop = pop_df['product_id'].head(10).tolist()
    
    # Matrix Prep
    user_item_train = train_events.pivot_table(index='user_id', columns='product_id', values='weight', aggfunc='sum', fill_value=0)
    user_ids = user_item_train.index.tolist()
    product_ids = user_item_train.columns.tolist()
    sparse_train = csr_matrix(user_item_train.values)
    
    # 1. Item-Item CF
    print("Training Item-Item CF...")
    item_sim = cosine_similarity(sparse_train.T)
    
    # 2. SVD
    print("Training SVD...")
    svd = TruncatedSVD(n_components=50, random_state=42)
    user_factors = svd.fit_transform(sparse_train)
    item_factors = svd.components_.T
    svd_preds = np.dot(user_factors, item_factors.T)
    
    # Evaluation
    test_user_groups = test_events.groupby('user_id')
    eval_users = list(test_user_groups.groups.keys())[:200] # Sample 200 users for eval
    
    metrics = {'Pop': [], 'CF': [], 'SVD': [], 'Hybrid': []}
    
    for uid in eval_users:
        if uid not in user_item_train.index: continue
        u_idx = user_ids.index(uid)
        actual = set(test_user_groups.get_group(uid)['product_id'])
        
        # Pop Preds
        pred_pop = top_10_pop
        
        # CF Preds
        u_vec = user_item_train.iloc[u_idx].values
        cf_scores = u_vec.dot(item_sim)
        cf_scores[u_vec > 0] = -1 # mask seen
        pred_cf = [product_ids[i] for i in np.argsort(cf_scores)[-10:][::-1]]
        
        # SVD Preds
        svd_scores = svd_preds[u_idx]
        svd_scores[u_vec > 0] = -1
        pred_svd = [product_ids[i] for i in np.argsort(svd_scores)[-10:][::-1]]
        
        # Hybrid (CF 0.5 + Pop 0.5)
        # Normalize scores for hybrid
        cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-9)
        # Simple pop rank score
        pop_scores = np.zeros(len(product_ids))
        for i, pid in enumerate(product_ids):
            if pid in pop_df['product_id'].values:
                pop_scores[i] = pop_df[pop_df['product_id'] == pid]['weight'].values[0]
        pop_norm = (pop_scores - pop_scores.min()) / (pop_scores.max() - pop_scores.min() + 1e-9)
        
        hybrid_scores = 0.7 * cf_norm + 0.3 * pop_norm
        hybrid_scores[u_vec > 0] = -1
        pred_hybrid = [product_ids[i] for i in np.argsort(hybrid_scores)[-10:][::-1]]
        
        metrics['Pop'].append(1 if actual.intersection(pred_pop) else 0)
        metrics['CF'].append(1 if actual.intersection(pred_cf) else 0)
        metrics['SVD'].append(1 if actual.intersection(pred_svd) else 0)
        metrics['Hybrid'].append(1 if actual.intersection(pred_hybrid) else 0)
        
    print(f"Results (Hit Rate @ 10):")
    for m, vals in metrics.items():
        print(f" - {m}: {np.mean(vals):.4f}")
        
    # Save Artifacts
    print("\n--- Step 3: Saving Best Artifacts ---")
    full_user_item = events.pivot_table(index='user_id', columns='product_id', values='weight', aggfunc='sum', fill_value=0)
    full_item_sim = cosine_similarity(csr_matrix(full_user_item.values).T)
    
    np.save(os.path.join(MODELS_DIR, 'item_similarity_v2.npy'), full_item_sim)
    with open(os.path.join(MODELS_DIR, 'matrix_metadata_v2.pkl'), 'wb') as f:
        pickle.dump({'cols': full_user_item.columns, 'index': full_user_item.index}, f)
    
    pop_df.to_csv(os.path.join(MODELS_DIR, 'popularity_v2.csv'), index=False)
    print("Done!")

if __name__ == '__main__':
    u, p, e = generate_data()
    train_and_evaluate(u, p, e)
