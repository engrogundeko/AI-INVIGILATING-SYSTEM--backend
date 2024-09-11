from pymongo import MongoClient, ReturnDocument


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

    def insert_many(self, documents):
        qs = self.collection.insert_many(documents)

    def find_one(self, query):
        return self.collection.find_one(query)

    def find(self, query, projection=None):
        return self.collection.find(query, projection=projection)

    def find_many(self, query=None):
        if query:
            return self.collection.find(query)
        return self.collection.find()

    def update_one(self, query, update):
        return self.collection.update_one(query, {"$set": update})

    def delete_one(self, query):
        return self.collection.delete_one(query)

    def get_next_sequence_value(self, sequence_name):
        return self.db.counters.find_one_and_update(
            {"_id": sequence_name},
            {"$inc": {"sequence_value": 1}},
            return_document=ReturnDocument.AFTER,
            upsert=True,
        )["sequence_value"]


userRespository = MongoDBRepository(collection_name="user")
examRespository = MongoDBRepository(collection_name="exam")
roomRespository = MongoDBRepository(collection_name="room")
courseRespository = MongoDBRepository(collection_name="course")
videoRecordingRespository = MongoDBRepository(collection_name="recording")
examAttedanceRespository = MongoDBRepository(collection_name="attendance")
notificationRespository = MongoDBRepository(collection_name="notification")
suspiciousReportRespository = MongoDBRepository(collection_name="suspicious")
detectionRespository = MongoDBRepository(collection_name="detections")
examRegistrationRespository = MongoDBRepository(collection_name="registration")
userRoleRepository = MongoDBRepository(collection_name="role")
examLocationRepo = MongoDBRepository(collection_name="examlocation")
studentDataRepo = MongoDBRepository(collection_name="exam_data")
sessionRepository = MongoDBRepository(collection_name="session")
departmentRepository = MongoDBRepository(collection_name="department")
facultyRepository = MongoDBRepository(collection_name="faculty")
if __name__ == "__main__":
    pass
# Initialize repository

# Example document
