const API_BASE = '/api';

// Format currency
const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(price);
};

// Generate Product Card HTML
const createProductCard = (product) => {
    const icon = product.category === 'Electronics' ? '📱' : 
                 product.category === 'Clothing' ? '👕' : 
                 product.category === 'Home & Kitchen' ? '🏠' : '📦';

    return `
        <div class="product-card">
            <div class="product-image">${icon}</div>
            <div class="product-info">
                <div class="product-category">${product.category}</div>
                <div class="product-title" title="${product.product_name}">${product.product_name}</div>
                <div class="product-rating">⭐ ${product.rating.toFixed(1)}</div>
                <div class="product-footer">
                    <div class="product-price">${formatPrice(product.price)}</div>
                </div>
                <button class="buy-btn" onclick="viewSimilar(${product.product_id})">View Similar</button>
            </div>
        </div>
    `;
};

// Render Products to a container
const renderProducts = (containerId, products) => {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    if (!products || products.length === 0) {
        container.innerHTML = '<p>No products found.</p>';
        return;
    }
    
    container.innerHTML = products.map(createProductCard).join('');
};

// Show/Hide loader
const toggleLoader = (show) => {
    const loader = document.getElementById('loader');
    if (loader) {
        loader.style.display = show ? 'block' : 'none';
    }
};

// Fetch Personalized Recommendations
const fetchRecommendations = async () => {
    const userIdInput = document.getElementById('user-id-input');
    if (!userIdInput) return;
    
    const userId = userIdInput.value;
    if (!userId) {
        alert('Please enter a User ID');
        return;
    }
    
    toggleLoader(true);
    try {
        const response = await fetch(`${API_BASE}/recommend/${userId}`);
        const data = await response.json();
        
        document.getElementById('algo-info').innerText = `Algorithm: ${data.algorithm}`;
        renderProducts('recommendation-grid', data.recommendations);
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        alert('Failed to fetch recommendations. Is the backend running?');
    } finally {
        toggleLoader(false);
    }
};

// Fetch Trending
const fetchTrending = async () => {
    toggleLoader(true);
    try {
        const response = await fetch(`${API_BASE}/trending`);
        const data = await response.json();
        renderProducts('trending-grid', data.recommendations);
    } catch (error) {
        console.error('Error fetching trending:', error);
    } finally {
        toggleLoader(false);
    }
};

// Fetch Dashboard Metrics
const fetchDashboardStats = async () => {
    try {
        const response = await fetch(`${API_BASE}/dashboard-data`);
        const data = await response.json();
        
        document.getElementById('total-users').innerText = data.total_users.toLocaleString();
        document.getElementById('total-revenue').innerText = formatPrice(data.total_revenue);
        document.getElementById('avg-ctr').innerText = `${data.avg_ctr}%`;
        document.getElementById('conversion').innerText = `${data.conversion_rate}%`;
    } catch (error) {
        console.error('Error fetching dashboard stats:', error);
    }
};

// View Similar (Redirect or load in place)
const viewSimilar = (productId) => {
    window.location.href = `similar.html?id=${productId}`;
};

// Auto-load logic based on page
document.addEventListener('DOMContentLoaded', () => {
    const path = window.location.pathname;
    
    if (path.includes('trending.html') || path === '/' || path.endsWith('/')) {
        const grid = document.getElementById('trending-grid');
        if (grid) fetchTrending();
    }
    
    if (path.includes('dashboard.html')) {
        fetchDashboardStats();
    }
    
    if (path.includes('similar.html')) {
        const urlParams = new URLSearchParams(window.location.search);
        const id = urlParams.get('id');
        if (id) {
            document.getElementById('similar-title').innerText = `Products similar to ID: ${id}`;
            fetchSimilar(id);
        }
    }
});

const fetchSimilar = async (productId) => {
    toggleLoader(true);
    try {
        const response = await fetch(`${API_BASE}/similar/${productId}`);
        const data = await response.json();
        renderProducts('similar-grid', data.recommendations);
    } catch (error) {
        console.error('Error fetching similar products:', error);
    } finally {
        toggleLoader(false);
    }
};
