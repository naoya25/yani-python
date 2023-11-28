from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
from recommend import get_yani
import os

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return "Flask running! Version 2.0"


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json()

    if "postData" not in data:
        return make_response(jsonify({"error": "Missing 'postData' in the request"}), 400)

    post_data = data["postData"]
    recommendation = get_yani(post_data)

    response_data = {"recommendation": recommendation}
    return jsonify(response_data)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port)
