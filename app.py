from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
from recomend import get_yani

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return "flask running!"


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    user_data = data.get("user_data")
    recommendation = get_yani(user_data)

    return make_response(jsonify({"recommendation": recommendation}))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
