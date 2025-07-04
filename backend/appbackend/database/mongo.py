import pymongo
from appbackend.config import Config

def get_mongo_collection():
    mongo_client = pymongo.MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_db = mongo_client["logoDB"]
    mongo_collection = mongo_db["logos"]
    return mongo_collection