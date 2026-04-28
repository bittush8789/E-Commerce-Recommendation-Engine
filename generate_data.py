import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

def generate_synthetic_data(num_rows=55000):
    print(f"Generating {num_rows} rows of synthetic e-commerce data...")
    
    # Constants
    num_users = 5000
    num_products = 1000
    categories = ['Electronics', 'Fashion', 'Home & Kitchen', 'Beauty', 'Sports', 'Books']
    brands = ['Apple', 'Samsung', 'Nike', 'Adidas', 'Sony', 'LG', 'Puma', 'Dell', 'HP', 'IKEA']
    
    # Create Directories
    os.makedirs('data', exist_ok=True)
    
    # Generate Products Metadata
    product_data = []
    for i in range(num_products):
        product_id = i + 1
        category = random.choice(categories)
        brand = random.choice(brands)
        price = round(random.uniform(10, 2000), 2)
        rating = round(random.uniform(3.0, 5.0), 1)
        product_data.append({
            'product_id': product_id,
            'category': category,
            'brand': brand,
            'price': price,
            'rating': rating,
            'product_name': f"{brand} {category} Item {product_id}"
        })
    products_df = pd.DataFrame(product_data)
    products_df.to_csv('data/products.csv', index=False)
    
    # Generate Interactions
    data = []
    start_date = datetime.now() - timedelta(days=90)
    
    for _ in range(num_rows):
        user_id = random.randint(1, num_users)
        product_id = random.randint(1, num_products)
        
        # Interaction types with weights
        interaction_type = random.choices(
            ['view', 'click', 'cart_addition', 'purchase'],
            weights=[0.6, 0.25, 0.1, 0.05],
            k=1
        )[0]
        
        # Weights for recommendation training
        weight = {'view': 1, 'click': 2, 'cart_addition': 5, 'purchase': 10}[interaction_type]
        
        timestamp = start_date + timedelta(
            days=random.randint(0, 89),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Search queries
        search_query = ""
        if random.random() < 0.2:
            prod_info = products_df[products_df['product_id'] == product_id].iloc[0]
            search_query = f"{prod_info['category']} {prod_info['brand']}"
            
        data.append({
            'user_id': user_id,
            'product_id': product_id,
            'interaction_type': interaction_type,
            'weight': weight,
            'timestamp': timestamp,
            'search_query': search_query,
            'wishlist': 1 if random.random() < 0.05 else 0
        })
        
    interactions_df = pd.DataFrame(data)
    
    # Add seasonal trends (e.g., more electronics in Nov/Dec)
    # Simple simulation: increase purchase weight for Electronics in "December"
    interactions_df['is_seasonal'] = interactions_df['timestamp'].apply(lambda x: 1 if x.month == 12 else 0)
    
    interactions_df.to_csv('data/interactions.csv', index=False)
    print("Data generation complete. Files saved in data/ directory.")

if __name__ == "__main__":
    generate_synthetic_data()
