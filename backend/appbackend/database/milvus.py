from pymilvus import connections, Collection
from appbackend.config import Config

def get_milvus_collection():
    connections.connect(uri=Config.ENDPOINT, token=Config.TOKEN)
    collection = Collection(name=Config.COLLECTION_NAME)
    return collection