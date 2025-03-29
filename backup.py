from flask import Flask, render_template, jsonify
import requests, numpy as np, time, threading, logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global cache variables
cached_history = None
cached_model = None  # Will store tuple: (model, scaler)
last_train_time = 0

def fetch_data():
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=200"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        history = [{"time": k[0], "price": float(k[1])} for k in data]
        return history
    except Exception as e:
        logging.error("Error fetching data: %s", e)
        return None

def compute_indicators(history):
    prices = np.array([d["price"] for d in history])
    sma5 = np.convolve(prices, np.ones(5) / 5, mode='valid')
    sma15 = np.convolve(prices, np.ones(15) / 15, mode='valid')
    sma5_full = np.concatenate((np.full(4, np.nan), sma5))
    sma15_full = np.concatenate((np.full(14, np.nan), sma15))
    return sma5_full, sma15_full, prices

def train_model(history):
    sma5, sma15, prices = compute_indicators(history)
    valid_idx = ~np.isnan(sma5) & ~np.isnan(sma15)
    X = np.column_stack((sma5[valid_idx], sma15[valid_idx]))
    prices_valid = prices[valid_idx]
    # Label: 1 if next candle's price is higher than current candle's, else 0
    y = (np.diff(prices_valid) > 0).astype(int)
    if len(X) <= 1:
        raise ValueError("Not enough data for training")
    X = X[:-1]  # Remove last row (no corresponding label)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y)
    return model, scaler

def update_model_cache():
    global cached_history, cached_model, last_train_time
    history = fetch_data()
    if history:
        try:
            model, scaler = train_model(history)
            cached_history = history
            cached_model = (model, scaler)
            last_train_time = time.time()
            logging.info("Model updated at %s", last_train_time)
        except Exception as e:
            logging.error("Error training model: %s", e)

def model_update_scheduler(interval=300):
    while True:
        update_model_cache()
        time.sleep(interval)

# Start the background thread to update model cache every 5 minutes
threading.Thread(target=model_update_scheduler, daemon=True).start()

def get_prediction():
    global cached_history, cached_model
    # If cache is empty, update now
    if cached_history is None or cached_model is None:
        update_model_cache()
    model, scaler = cached_model
    sma5, sma15, _ = compute_indicators(cached_history)
    last_features = np.array([[sma5[-1], sma15[-1]]])
    last_features_scaled = scaler.transform(last_features)
    pred = model.predict(last_features_scaled)[0]
    confidence = model.predict_proba(last_features_scaled)[0][pred] * 100  # as percentage
    prediction = "🟢 ↑" if pred == 1 else "🔴 ↓"
    return cached_history, prediction, round(confidence, 2)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/api/historical')
def historical_api():
    history, prediction, confidence = get_prediction()
    return jsonify({
        "history": history,
        "prediction": prediction,
        "confidence": f"{confidence}%"
    })

if __name__ == '__main__':
    app.run(debug=True)
