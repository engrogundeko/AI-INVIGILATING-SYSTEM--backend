from pymongo import MongoClient


class MongoDBRepository:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        db_name: str = "AIReady",
        collection_name: str = None,
    ):
        self.client = MongoClient(host, port)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def insert_one(self, document):
        qs = self.collection.insert_one(document)
        return self.find_one({"_id": qs.inserted_id})

    def find_one(self, query):
        return self.collection.find_one(query)

    def find_many(self, query=None):
        if query:
            return self.collection.find(query)
        return self.collection.find()

    def update_one(self, query, update):
        return self.collection.update_one(query, {"$set": update})

    def delete_one(self, query):
        return self.collection.delete_one(query)


userRespository = MongoDBRepository(collection_name="user")
examRespository = MongoDBRepository(collection_name="exam")
roomRespository = MongoDBRepository(collection_name="room")
courseRespository = MongoDBRepository(collection_name="course")
examAttedanceRespository = MongoDBRepository(collection_name="attendance")
examRegistrationRespository = MongoDBRepository(collection_name="registration")

if __name__ == "__main__":
    pass
