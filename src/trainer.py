import pandas as pd
import pickle
import os
from data_generator import generate_synthetic_data
from recommender import RecommenderModels, calculate_hit_rate
from sklearn.model_selection import train_test_split

def run_pipeline():
    # 1. Data Generation
    users, products, interactions = generate_synthetic_data()
    
    # 2. Split
    train_df, test_df = train_test_split(interactions, test_size=0.2, random_state=42)
    
    # 3. Model Prep
    model_suite = RecommenderModels(train_df)
    
    # 4. Comparison
    print("Evaluating Models...")
    hr_pop = calculate_hit_rate(lambda u, n: model_suite.get_popularity_recommendations(n), test_df)
    print(f"Popularity Hit Rate @ 10: {hr_pop:.4f}")
    
    hr_cf = calculate_hit_rate(model_suite.get_cf_recommendations, test_df)
    print(f"Item-Item CF Hit Rate @ 10: {hr_cf:.4f}")
    
    hr_svd = calculate_hit_rate(model_suite.get_svd_recommendations, test_df)
    print(f"SVD Hit Rate @ 10: {hr_svd:.4f}")
    
    # 5. Selection
    best_hr = max(hr_pop, hr_cf, hr_svd)
    if best_hr == hr_cf:
        best_type = "Collaborative Filtering"
        # Re-train on full data
        final_model = RecommenderModels(interactions)
    elif best_hr == hr_svd:
        best_type = "Matrix Factorization (SVD)"
        final_model = RecommenderModels(interactions)
    else:
        best_type = "Popularity"
        final_model = RecommenderModels(interactions)
        
    print(f"Winner: {best_type} ({best_hr:.4f})")
    
    # 6. Save Artifacts
    os.makedirs('models', exist_ok=True)
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    
    with open('models/metadata.pkl', 'wb') as f:
        pickle.dump({
            'best_type': best_type,
            'hit_rate': best_hr,
            'user_ids': final_model.user_ids,
            'product_ids': final_model.product_ids
        }, f)
    
    print("Project upgraded and models saved in 'models/' directory.")

if __name__ == "__main__":
    run_pipeline()
