from flask import Flask, render_template, jsonify, request
import requests, numpy as np, time, threading, logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Notification settings
NTFY_TOPIC = "project_ts"  # Change this to your topic

def send_notification(message):
    print(f"Sending notification: {message}")  # Debug
    try:
        response = requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=message.encode("utf-8"))
        logging.info("Notification response: %s", response.text)
    except Exception as e:
        logging.error("Error sending notification: %s", e)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global cache for model data
cached_history = None
cached_model = None  # (model, scaler)
last_train_time = 0

# Global variables for alerts:
alert_range = None       # Tuple: (lower, upper) for range alerts
single_alert = None      # Float: target price for single alert mode

def fetch_data():
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=200"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        history = [{"time": k[0], "price": float(k[1])} for k in data]
        return history
    except Exception as e:
        logging.error("Error fetching historical data: %s", e)
        return None

def compute_indicators(history):
    prices = np.array([d["price"] for d in history])
    sma5 = np.convolve(prices, np.ones(5)/5, mode='valid')
    sma15 = np.convolve(prices, np.ones(15)/15, mode='valid')
    sma5_full = np.concatenate((np.full(4, np.nan), sma5))
    sma15_full = np.concatenate((np.full(14, np.nan), sma15))
    return sma5_full, sma15_full, prices

def train_model(history):
    sma5, sma15, prices = compute_indicators(history)
    valid_idx = ~np.isnan(sma5) & ~np.isnan(sma15)
    X = np.column_stack((sma5[valid_idx], sma15[valid_idx]))
    prices_valid = prices[valid_idx]
    y = (np.diff(prices_valid) > 0).astype(int)
    if len(X) <= 1:
        raise ValueError("Not enough data for training")
    X = X[:-1]
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

threading.Thread(target=model_update_scheduler, daemon=True).start()

def get_prediction():
    global cached_history, cached_model
    if cached_history is None or cached_model is None:
        update_model_cache()
    model, scaler = cached_model
    sma5, sma15, _ = compute_indicators(cached_history)
    last_features = np.array([[sma5[-1], sma15[-1]]])
    last_features_scaled = scaler.transform(last_features)
    pred = model.predict(last_features_scaled)[0]
    confidence = model.predict_proba(last_features_scaled)[0][pred] * 100
    prediction = "🟢 ↑" if pred == 1 else "🔴 ↓"
    return cached_history, prediction, round(confidence, 2)

def fetch_live_price():
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return float(data["price"])
    except Exception as e:
        logging.error("Error fetching live price: %s", e)
        return None

def price_monitor():
    global alert_range, single_alert
    range_notified = False
    single_notified = False

    while True:
        current_price = fetch_live_price()
        logging.info(f"Live BTC price: {current_price}")

        # Check range alert:
        if current_price and alert_range is not None:
            lower, upper = alert_range
            logging.info(f"Range alert set: {lower} - {upper}")
            if lower <= current_price <= upper and not range_notified:
                message = f"🚨 BTC Alert: Price reached ${current_price:.2f} (Range: ${lower:.2f}-${upper:.2f})"
                send_notification(message)
                logging.info(message)
                range_notified = True
            elif current_price < lower or current_price > upper:
                range_notified = False

        # Check single alert:
        if current_price and single_alert is not None:
            logging.info(f"Single alert target: {single_alert}")
            if current_price >= single_alert and not single_notified:
                message = f"🚨 BTC Single Alert: Price reached ${current_price:.2f} (Target: ${single_alert:.2f})"
                send_notification(message)
                logging.info(message)
                single_notified = True
            elif current_price < single_alert:
                single_notified = False

        time.sleep(3)

threading.Thread(target=price_monitor, daemon=True).start()

@app.route("/")
def index():
    return render_template("index2.html", alert_success=False)

@app.route("/api/historical")
def historical_api():
    history, prediction, confidence = get_prediction()
    range_str = f"{alert_range[0]:.2f} - {alert_range[1]:.2f}" if alert_range is not None else "None"
    single_str = f"{single_alert:.2f}" if single_alert is not None else "None"
    return jsonify({
        "history": history,
        "prediction": prediction,
        "confidence": f"{confidence}%",
        "target_price": range_str,
        "single_alert": single_str
    })

@app.route("/set_alert", methods=["POST"])
def set_alert():
    global alert_range
    try:
        lower = float(request.form.get("lower", 0))
        upper = float(request.form.get("upper", 0))
        if lower >= upper:
            return render_template("index2.html", alert_success=False, alert_message="❌ Invalid range! Lower must be less than Upper.")
        alert_range = (lower, upper)
        logging.info("✅ Alert range set: %s - %s", lower, upper)
        print(f"🚀 Alert Set: BTC will notify when price is between ${lower:.2f} and ${upper:.2f}")
        return render_template("index2.html", alert_success=True, alert_message=f"✅ Alert set for range: ${lower:.2f} - ${upper:.2f}")
    except ValueError:
        logging.error("❌ Error: Non-numeric values provided.")
        return render_template("index2.html", alert_success=False, alert_message="❌ Please enter numeric values only.")
    except Exception as e:
        logging.error("❌ Unexpected error: %s", e)
        return render_template("index2.html", alert_success=False, alert_message="❌ Something went wrong. Try again.")

@app.route("/set_alert_single", methods=["POST"])
def set_alert_single():
    global single_alert
    try:
        price = float(request.form.get("price", 0))
        single_alert = price
        logging.info("✅ Single alert set at: %s", price)
        print(f"🚀 Single Alert Set: BTC will notify when price reaches ${price:.2f}")
        return render_template("index2.html", alert_success=True, alert_message=f"✅ Single alert set for: ${price:.2f}")
    except ValueError:
        logging.error("❌ Error: Non-numeric value provided for single alert.")
        return render_template("index2.html", alert_success=False, alert_message="❌ Please enter a numeric value.")
    except Exception as e:
        logging.error("❌ Unexpected error: %s", e)
        return render_template("index2.html", alert_success=False, alert_message="❌ Something went wrong. Try again.")

if __name__ == "__main__":
    app.run(debug=True)
