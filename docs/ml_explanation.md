# ML Engineering Documentation

## 1. Why this approach?
We moved from a simple "count-based" model to an **Implicit Feedback Weighting** model. In e-commerce, user signals (view vs purchase) are not equal. By assigning weights, we force the model to prioritize conversion-oriented signals.

## 2. Preprocessing & Data Cleaning
- **Duplicates**: Handled by aggregating interactions into a User-Item weight matrix.
- **Sparsity**: Addressed using `csr_matrix` and `TruncatedSVD` which performs dimensionality reduction.
- **Cold Start**: Handled by falling back to the Global Popularity model for new users.

## 3. Algorithms Explained
- **Popularity**: High-baseline model. Good for general trends and cold-start.
- **Collaborative Filtering (Item-Item)**: Uses cosine similarity between item vectors to find products that are "functionally similar" based on user behavior.
- **SVD (Matrix Factorization)**: Decomposes the interaction matrix into latent factors for users and items, capturing hidden patterns in behavior.

## 4. Evaluation Strategy
We use **Hit Rate @ 10**. This measures if the actual product the user interacted with in the test set appeared in our top 10 list. This is more aligned with business goals than simple RMSE/MAE for a ranking system.

## 5. Deployment
The model is serialized using `pickle`. The FastAPI backend (`app.py`) loads these files at startup to provide low-latency predictions.
