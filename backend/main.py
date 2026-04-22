from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import RecommendationResponse, ABTestRequest, ABTestResponse, DashboardStats
from recommender import recommender_system
from utils import generate_mock_ab_metrics, get_dashboard_metrics

app = FastAPI(title="E-Commerce Recommendation Engine API", version="1.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: int):
    try:
        recs = recommender_system.get_personalized(user_id)
        return {"user_id": user_id, "recommendations": recs, "algorithm": "hybrid_v1"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/similar/{product_id}", response_model=RecommendationResponse)
def get_similar(product_id: int):
    try:
        recs = recommender_system.get_similar_products(product_id)
        return {"user_id": None, "recommendations": recs, "algorithm": "item_item_collaborative"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trending", response_model=RecommendationResponse)
def get_trending():
    try:
        recs = recommender_system.get_trending()
        return {"user_id": None, "recommendations": recs, "algorithm": "popularity"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ab-test", response_model=ABTestResponse)
def ab_test(request: ABTestRequest):
    metrics = generate_mock_ab_metrics()
    winner = request.model_b if metrics['model_b_conversion'] > metrics['model_a_conversion'] else request.model_a
    return {
        "winner": winner,
        "model_a_ctr": metrics['model_a_ctr'],
        "model_b_ctr": metrics['model_b_ctr'],
        "model_a_conversion": metrics['model_a_conversion'],
        "model_b_conversion": metrics['model_b_conversion']
    }

@app.post("/bulk-score")
def bulk_score(data: list):
    return recommender_system.bulk_score(data)

@app.get("/dashboard-data", response_model=DashboardStats)
def get_dashboard_data():
    return get_dashboard_metrics()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
