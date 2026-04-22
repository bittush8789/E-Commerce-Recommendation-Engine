from pydantic import BaseModel
from typing import List, Optional

class Product(BaseModel):
    product_id: int
    product_name: str
    category: str
    price: float
    rating: float
    similarity_score: Optional[float] = None
    buy_link: Optional[str] = None

class RecommendationResponse(BaseModel):
    user_id: Optional[int] = None
    recommendations: List[Product]
    algorithm: str

class ABTestRequest(BaseModel):
    model_a: str
    model_b: str
    user_id: int

class ABTestResponse(BaseModel):
    winner: str
    model_a_ctr: float
    model_b_ctr: float
    model_a_conversion: float
    model_b_conversion: float

class DashboardStats(BaseModel):
    total_users: int
    total_products: int
    total_revenue: float
    avg_ctr: float
    conversion_rate: float
