# 🛒 Next-Gen E-Commerce Recommendation Engine

![Project Banner](docs/assets/banner.png)

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An end-to-end, production-grade recommendation system leveraging hybrid filtering techniques to deliver hyper-personalized shopping experiences. This project implements advanced ML pipelines, a high-performance FastAPI backend, and an immersive real-time dashboard.

---

## 🌟 Key Features

- **Hybrid Recommendation Engine**: Combines **Collaborative Filtering**, **Matrix Factorization (SVD)**, and **Popularity-based Ranking** to solve the cold-start problem and deliver accurate results.
- **Implicit Feedback Modeling**: Transforms user interactions (views, clicks, purchases) into a weighted scoring system for deeper behavioral understanding.
- **Automated MLOps Pipeline**: Features a robust training loop with automated model selection based on **Hit Rate @ K** metrics.
- **Real-Time API**: High-concurrency FastAPI backend with modular service architecture.
- **Immersive Frontend**: A sleek, dark-mode dashboard built with Vanilla JS/CSS for real-time visualization of recommendations.
- **Data-Driven Insights**: Built-in data generator for creating massive synthetic datasets that mimic real-world e-commerce behavior.

---

## 📂 Architecture & Structure

```bash
ecommerce-recommendation-engine/
├── data/               # Persistent storage for CSV datasets
├── models/             # Serialized ML models and versioned metadata
├── src/                # Core Python source logic
│   ├── data_generator.py # Synthetic data generation engine
│   ├── recommender.py    # Multi-model algorithm implementations
│   └── trainer.py        # Pipeline orchestration & evaluation
├── frontend/           # Modern Vanilla JS/CSS web application
│   ├── css/            # Custom UI design system
│   ├── js/             # Frontend logic & API integration
│   └── *.html          # Responsive landing & recommendation pages
├── docs/               # Technical documentation & assets
├── app.py              # Application entry point (FastAPI)
├── run_advanced_pipeline.py # End-to-end automation script
└── requirements.txt    # Project dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ecommerce-recommendation-engine.git
   cd ecommerce-recommendation-engine
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the System

1. **Bootstrap the Project**: Generate data and train the initial model.
   ```bash
   python run_advanced_pipeline.py
   ```

2. **Launch the API & Dashboard**:
   ```bash
   python -m uvicorn app:app --port 8000 --reload
   ```
   *Access the UI at [http://localhost:8000](http://localhost:8000)*

---

## 🧠 Machine Learning Details

### Model Evaluation
We prioritize accuracy and relevance using the **Hit Rate @ 10** metric. Our pipeline evaluates:
- **SVD (Singular Value Decomposition)**: Best for capturing latent features in sparse matrices.
- **Collaborative Filtering**: Leverages user-item similarity scores.
- **Popularity Baseline**: Ensures fallback recommendations for new users.

### API Endpoints
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/recommend/{user_id}` | `GET` | Get personalized recommendations for a user |
| `/api/similar/{product_id}` | `GET` | Find products similar to a given ID |
| `/api/trending` | `GET` | Fetch globally trending products |
| `/api/status` | `GET` | System health and model metadata |

---

## 🛠️ Built With

- **Backend**: FastAPI, Uvicorn, Python
- **ML/Math**: NumPy, Pandas, Scikit-Learn, Surprise
- **Frontend**: HTML5, CSS3 (Glassmorphism), Vanilla JavaScript
- **Serialization**: Pickle

---

## 📝 Roadmap
- [ ] Implement Deep Learning-based ranking (NCF).
- [ ] Add real-time user behavior tracking via WebSockets.
- [ ] Integrate a Vector Database (FAISS/ChromaDB) for faster similarity search.
- [ ] Containerize the entire stack with Docker Compose.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---
**Developed with ❤️ for the AI community.**
