import pandas as pd
import os
import random

# Use absolute paths or paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Load main datasets into memory for fast querying
print("Loading data into memory for backend...")
products_df = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
users_df = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))
# We don't load all events/orders here to save memory, only products and users

def get_product_details(product_ids):
    """Retrieve product details given a list of product IDs."""
    filtered = products_df[products_df['product_id'].isin(product_ids)]
    # Keep the order of the product_ids
    filtered['product_id'] = pd.Categorical(filtered['product_id'], categories=product_ids, ordered=True)
    filtered = filtered.sort_values('product_id')
    return filtered.to_dict('records')

def get_user_details(user_id):
    user = users_df[users_df['user_id'] == user_id]
    if user.empty:
        return None
    return user.iloc[0].to_dict()

def get_random_products(n=10):
    return products_df.sample(n).to_dict('records')

def generate_mock_ab_metrics():
    # Simulate realistic A/B testing metrics
    return {
        "model_a_ctr": round(random.uniform(2.5, 4.0), 2),
        "model_b_ctr": round(random.uniform(3.5, 5.5), 2),
        "model_a_conversion": round(random.uniform(0.5, 1.2), 2),
        "model_b_conversion": round(random.uniform(1.0, 2.5), 2)
    }

def get_dashboard_metrics():
    # In a real app this would query the DB. Here we mock some realistic aggregations based on our dataset sizes.
    return {
        "total_users": len(users_df),
        "total_products": len(products_df),
        "total_revenue": 456230.50, # Mocked value
        "avg_ctr": 3.42,
        "conversion_rate": 1.25
    }
