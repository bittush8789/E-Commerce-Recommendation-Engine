import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_data(data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)
    np.random.seed(42)
    
    # 1. Users
    num_users = 2000
    users = pd.DataFrame({
        'user_id': range(1, num_users + 1),
        'age': np.random.randint(18, 70, size=num_users),
        'gender': np.random.choice(['M', 'F', 'O'], size=num_users),
        'preferred_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], size=num_users),
        'join_date': [datetime.now() - timedelta(days=np.random.randint(0, 730)) for _ in range(num_users)]
    })
    users.to_csv(os.path.join(data_dir, 'users.csv'), index=False)
    
    # 2. Products
    num_products = 500
    categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports']
    products = pd.DataFrame({
        'product_id': range(1, num_products + 1),
        'category': np.random.choice(categories, size=num_products),
        'price': np.round(np.random.uniform(5, 1000, size=num_products), 2),
        'rating': np.round(np.random.uniform(3, 5, size=num_products), 1),
        'stock': np.random.randint(0, 100, size=num_products)
    })
    products.to_csv(os.path.join(data_dir, 'products.csv'), index=False)
    
    # 3. Interactions
    num_interactions = 40000
    interactions = []
    for _ in range(num_interactions):
        uid = np.random.randint(1, num_users + 1)
        pid = np.random.randint(1, num_products + 1)
        # Probability based on user preference
        u_pref = users.loc[uid-1, 'preferred_category']
        p_cat = products.loc[pid-1, 'category']
        
        # Bias towards preferred category
        if u_pref == p_cat and np.random.rand() < 0.4:
            etype = np.random.choice(['view', 'click', 'add_to_cart', 'purchase'], p=[0.4, 0.3, 0.2, 0.1])
        else:
            etype = np.random.choice(['view', 'click', 'add_to_cart', 'purchase'], p=[0.7, 0.2, 0.08, 0.02])
            
        interactions.append([uid, pid, etype, datetime.now() - timedelta(days=np.random.randint(0, 365))])
        
    df_inter = pd.DataFrame(interactions, columns=['user_id', 'product_id', 'event_type', 'timestamp'])
    # Weighted interactions for implicit feedback
    weights = {'view': 1, 'click': 2, 'add_to_cart': 5, 'purchase': 10}
    df_inter['weight'] = df_inter['event_type'].map(weights)
    df_inter.to_csv(os.path.join(data_dir, 'interactions.csv'), index=False)
    
    print(f"Data generated: {num_users} users, {num_products} products, {num_interactions} interactions.")
    return users, products, df_inter

if __name__ == "__main__":
    generate_synthetic_data()
