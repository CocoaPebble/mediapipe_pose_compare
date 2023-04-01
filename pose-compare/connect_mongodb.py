import pymongo

client = pymongo.MongoClient(
    "mongodb+srv://YingzhouJiang:Jyz1996!@cluster0.zkmuv24.mongodb.net/?retryWrites=true&w=majority",
    serverSelectionTimeoutMS=5000,
)

db = client["ufit_test"]
exercises = db["exercises"]
