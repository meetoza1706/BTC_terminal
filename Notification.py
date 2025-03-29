from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

NTFY_TOPIC = "project_ts"  # Change this to your topic

def send_notification(message):
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    requests.post(url, data=message)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        message = request.form["message"]
        send_notification(message)
        return render_template("index.html", success=True)
    return render_template("index.html", success=False)

if __name__ == "__main__":
    app.run(debug=True)
