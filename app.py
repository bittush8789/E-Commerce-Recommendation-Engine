from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI(title="E-Commerce MLOps Recommendation Engine")

# Metrics
REQUEST_COUNT = Counter('request_count', 'Total Request Count', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency', ['endpoint'])
REC_COUNT = Counter('recommendation_count', 'Total recommendations served')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model and Data
MODEL_PATH = os.getenv('MODEL_PATH', 'models/model.pkl')
PRODUCTS_PATH = 'data/products.csv'

model = None
products_df = None

@app.on_event("startup")
def startup_event():
    global model, products_df
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
        if os.path.exists(PRODUCTS_PATH):
            products_df = pd.read_csv(PRODUCTS_PATH)
            print("Products data loaded.")
    except Exception as e:
        print(f"Error loading resources: {e}")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/api/recommend/{user_id}")
async def recommend(user_id: int, n: int = 10):
    REQUEST_COUNT.labels(method='GET', endpoint='/recommend', http_status=200).inc()
    with REQUEST_LATENCY.labels(endpoint='/recommend').time():
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        rec_ids = model.get_recommendations(user_id, n=n)
        recommendations = products_df[products_df['product_id'].isin(rec_ids)].to_dict('records')
        
        REC_COUNT.inc(len(recommendations))
        return {"user_id": user_id, "recommendations": recommendations}

@app.get("/api/similar/{product_id}")
async def similar(product_id: int, n: int = 10):
    if products_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Simple similarity based on category
    target_product = products_df[products_df['product_id'] == product_id]
    if target_product.empty:
        raise HTTPException(status_code=404, detail="Product not found")
    
    category = target_product.iloc[0]['category']
    similar_items = products_df[products_df['category'] == category].head(n).to_dict('records')
    return {"product_id": product_id, "similar_products": similar_items}

@app.get("/api/trending")
async def trending(n: int = 10):
    if products_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Return top rated as trending
    trending_items = products_df.sort_values(by='rating', ascending=False).head(n).to_dict('records')
    return {"trending": trending_items}

# Serve Frontend
if os.path.exists('templates'):
    app.mount("/", StaticFiles(directory="templates", html=True), name="frontend")
    # If static directory exists, mount it
    if os.path.exists('static'):
        app.mount("/static", StaticFiles(directory="static"), name="static")
