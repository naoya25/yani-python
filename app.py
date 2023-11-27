from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
from recommend import get_yani
import os

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return "flask running! ver 2.0"


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    post_data = data.get("postData")
    recommendation = get_yani(post_data)

    return make_response(jsonify({"recommendation": recommendation}))


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)))
