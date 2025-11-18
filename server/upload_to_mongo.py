import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
mongo_url = os.getenv("MONGO_URL")
df = pd.read_csv("server/synthetic_students_realistic.csv")

client = MongoClient(mongo_url) 
db = client["student_db"]  
collection = db["students"] 

collection.delete_many({})

records = df.to_dict(orient="records")
collection.insert_many(records)

print(f"Inserted {len(records)} records into MongoDB!")
