import pandas as pd
import pickle
import os
import random
from utils import get_product_details, get_user_details, get_random_products

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

class Recommender:
    def __init__(self):
        print("Initializing Recommender...")
        self.load_models()
        
    def load_models(self):
        # Load Trending Model
        self.trending_df = pd.read_csv(os.path.join(MODELS_DIR, 'popularity_model.csv'))
        
        # Load FBT Model
        with open(os.path.join(MODELS_DIR, 'fbt_model.pkl'), 'rb') as f:
            self.fbt_model = pickle.load(f)
            
        # Load Category Model
        with open(os.path.join(MODELS_DIR, 'category_model.pkl'), 'rb') as f:
            self.category_model = pickle.load(f)

    def get_trending(self, limit=10):
        top_products = self.trending_df.head(limit).to_dict('records')
        return top_products
        
    def get_personalized(self, user_id, limit=10):
        user = get_user_details(user_id)
        if not user:
            # Cold start user: Return trending
            return self.get_trending(limit)
            
        pref_cat = user.get('preferred_category')
        if pref_cat in self.category_model:
            # Return mix of preferred category and trending
            cat_product_ids = self.category_model[pref_cat][:limit//2]
            trending_products = self.trending_df.head(limit).to_dict('records')
            
            cat_products = get_product_details(cat_product_ids)
            
            # Combine and deduplicate
            combined = cat_products + trending_products
            seen = set()
            result = []
            for p in combined:
                if p['product_id'] not in seen:
                    seen.add(p['product_id'])
                    result.append(p)
                if len(result) >= limit:
                    break
            return result
            
        return self.get_trending(limit)

    def get_similar_products(self, product_id, limit=10):
        # First check FBT
        if product_id in self.fbt_model:
            fbt_ids = self.fbt_model[product_id][:limit]
            fbt_products = get_product_details(fbt_ids)
            if len(fbt_products) >= limit:
                return fbt_products
            
            # Need more? Pad with random products from same category
            # In a real system, we'd use item similarity matrix
            result = fbt_products
            needed = limit - len(result)
            random_pad = get_random_products(needed)
            return result + random_pad
            
        return get_random_products(limit)

    def bulk_score(self, request_data):
        return {"status": "success", "scored": len(request_data)}

recommender_system = Recommender()
