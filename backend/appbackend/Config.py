import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "../dataset/Dataset_simplified")
    TEMP_DIR = os.path.join(BASE_DIR, "../temp")
    PRETRAINED_PATH = os.path.join(BASE_DIR, "../pretrained/hierarchical_ordered.pth.tar")
    ENDPOINT = "https://in03-754f3454a65e40f.serverless.gcp-us-west1.cloud.zilliz.com"
    TOKEN = "2b830a69fb087e580f904877ff816ff1477e67a38c091fc6b8c9c75d3992a458cc2deb681d3ae18dd91900855e1c538013080bf5"
    COLLECTION_NAME = "fyp_project"
    MONGO_URI = "mongodb+srv://hiru23anjalee:p24gomepFiz7R9HB@cluster0.kt9cubt.mongodb.net/?retryWrites=true&w=majority"