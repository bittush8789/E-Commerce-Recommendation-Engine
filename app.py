from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pickle
import os
import pandas as pd
from typing import List, Dict
import traceback

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from recommender import RecommenderModels
except ImportError:
    from src.recommender import RecommenderModels

app = FastAPI(title="Advanced E-Commerce Recommendation API")

# Add CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store models
model = None
metadata = None
products_df = None

@app.on_event("startup")
def load_model():
    global model, metadata, products_df
    try:
        model_path = os.path.join('models', 'best_model.pkl')
        meta_path = os.path.join('models', 'metadata.pkl')
        data_path = os.path.join('data', 'products.csv')

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
        
        if os.path.exists(data_path):
            products_df = pd.read_csv(data_path)
            if 'product_name' not in products_df.columns:
                products_df['product_name'] = products_df['category'] + " Item #" + products_df['product_id'].astype(str)
        
        if metadata:
            print(f"Successfully loaded {metadata.get('best_type', 'Unknown')} model.")
        else:
            print("Metadata not found, but model might be loaded.")
            
    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()

# API Routes prefixed with /api
@app.get("/api/status")
def get_status():
    return {
        "status": "running", 
        "model": metadata.get('best_type') if metadata else "Not Loaded",
        "has_model": model is not None,
        "has_products": products_df is not None
    }

@app.get("/api/recommend/{user_id}")
def get_recommendations(user_id: int, n: int = 10):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    best_type = metadata.get('best_type', "Popularity")
    
    if best_type == "Matrix Factorization (SVD)":
        rec_ids = model.get_svd_recommendations(user_id, n)
    elif best_type == "Collaborative Filtering":
        rec_ids = model.get_cf_recommendations(user_id, n)
    else:
        rec_ids = model.get_popularity_recommendations(n)
        
    recs = products_df[products_df['product_id'].isin(rec_ids)].to_dict('records')
    return {
        "user_id": user_id, 
        "recommendations": recs,
        "algorithm": best_type
    }

@app.get("/api/similar/{product_id}")
def get_similar(product_id: int, n: int = 10):
    if products_df is None:
        raise HTTPException(status_code=500, detail="Product data not loaded")
        
    p_info = products_df[products_df['product_id'] == product_id]
    if p_info.empty:
        raise HTTPException(status_code=404, detail="Product not found")
    
    cat = p_info.iloc[0]['category']
    similar_recs = products_df[products_df['category'] == cat].head(n).to_dict('records')
    return {"recommendations": similar_recs}

@app.get("/api/trending")
def get_trending(n: int = 10):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    rec_ids = model.get_popularity_recommendations(n)
    recs = products_df[products_df['product_id'].isin(rec_ids)].to_dict('records')
    return {"recommendations": recs}

@app.get("/api/dashboard-data")
def get_dashboard():
    if not metadata:
        return {"error": "Metadata not loaded"}
    return {
        "total_users": len(metadata.get('user_ids', [])),
        "total_revenue": 1250000.50,
        "avg_ctr": 4.2,
        "conversion_rate": 2.1
    }

@app.get("/api/health")
def health():
    return {"status": "ok"}

# Serve the frontend
# Note: Mount this LAST so it doesn't shadow the /api routes
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

