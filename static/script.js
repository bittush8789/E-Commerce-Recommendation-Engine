async function getRecommendations() {
    const userId = document.getElementById('userId').value;
    const recGrid = document.getElementById('recGrid');
    
    recGrid.innerHTML = '<div class="loading">Fetching Intelligence...</div>';
    
    try {
        const response = await fetch(`/api/recommend/${userId}`);
        const data = await response.json();
        
        recGrid.innerHTML = '';
        data.recommendations.forEach(product => {
            recGrid.appendChild(createProductCard(product));
        });
    } catch (error) {
        console.error('Error:', error);
        recGrid.innerHTML = '<div class="error">System Offline or User Not Found</div>';
    }
}

async function getTrending() {
    const trendGrid = document.getElementById('trendGrid');
    try {
        const response = await fetch('/api/trending');
        const data = await response.json();
        
        trendGrid.innerHTML = '';
        data.trending.forEach(product => {
            trendGrid.appendChild(createProductCard(product));
        });
    } catch (error) {
        console.error('Error:', error);
    }
}

function createProductCard(product) {
    const card = document.createElement('div');
    card.className = 'product-card';
    card.innerHTML = `
        <div class="cat">${product.category} | ${product.brand}</div>
        <h4>${product.product_name}</h4>
        <div class="footer">
            <div class="price">$${product.price}</div>
            <div class="rating">★ ${product.rating}</div>
        </div>
    `;
    card.onclick = () => getSimilar(product.product_id);
    return card;
}

async function getSimilar(productId) {
    const recGrid = document.getElementById('recGrid');
    recGrid.innerHTML = '<div class="loading">Finding Similar Gems...</div>';
    
    try {
        const response = await fetch(`/api/similar/${productId}`);
        const data = await response.json();
        
        recGrid.innerHTML = '';
        data.similar_products.forEach(product => {
            recGrid.appendChild(createProductCard(product));
        });
        
        window.scrollTo({ top: 500, behavior: 'smooth' });
    } catch (error) {
        console.error('Error:', error);
    }
}

// Initial Load
document.addEventListener('DOMContentLoaded', () => {
    getRecommendations();
    getTrending();
});
