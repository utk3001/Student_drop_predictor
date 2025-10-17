from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_student import predict_student_outcome
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Access your MongoDB URI
mongo_url = os.getenv("MONGO_URL")

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "Student Outcome Prediction API is running." 

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.json  # JSON with 36 features
    try:
        result = predict_student_outcome(input_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

client = MongoClient(mongo_url)
db = client["student_db"]
collection = db["students"]

@app.route("/get_student/<roll_no>", methods=["GET"])
def get_student(roll_no):
    try:
        roll_no = int(roll_no)
    except ValueError:
        return jsonify({"error": "Invalid Roll_No"}), 400

    student = collection.find_one({"Roll_No": roll_no})
    if student:
        # Remove MongoDB internal _id before returning
        student.pop("_id", None)
        return jsonify(student)
    else:
        return jsonify({"error": "Student not found"}), 404


if __name__ == "__main__":
    app.run(port=8000, debug=True)


