import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Access your MongoDB URI
mongo_url = os.getenv("MONGO_URL")


# Load CSV
df = pd.read_csv("server/synthetic_students_realistic.csv")

# Connect to MongoDB
client = MongoClient(mongo_url)  # Change URI if using cloud MongoDB
db = client["student_db"]  # Database name
collection = db["students"]  # Collection name

# Optional: Clear collection first
collection.delete_many({})

# Convert dataframe to dictionary and insert
records = df.to_dict(orient="records")
collection.insert_many(records)

print(f"Inserted {len(records)} records into MongoDB!")
