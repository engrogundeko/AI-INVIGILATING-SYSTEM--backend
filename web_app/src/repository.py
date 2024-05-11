from pymongo import MongoClient


class MongoDBRepository:
    def __init__(self, host="localhost", port=27017, db_name="AIReady"):

        self.client = MongoClient(host, port)
        self.db = self.client[db_name]

    def insert_one(self, collection_name, document):
        collection = self.db[collection_name]
        return collection.insert_one(document)

    def find_one(self, collection_name, query):
        collection = self.db[collection_name]
        return collection.find_one(query)

    def find_many(self, collection_name, query = None):
        collection = self.db[collection_name]
        if query:
            return collection.find(query)
        return collection.find()

    def update_one(self, collection_name, query, update):
        collection = self.db[collection_name]
        return collection.update_one(query, {"$set": update})

    def delete_one(self, collection_name, query):
        collection = self.db[collection_name]
        return collection.delete_one(query)


repository = MongoDBRepository()
# Example usage:
if __name__ == "__main__":
    # Initialize repository

    # Example document
    document = {"name": "John Doe", "age": 30, "email": "john.doe@example.com"}

    # Insert document into collection
    repository.insert_one("users", document)

    # Find document by query
    result = repository.find_one("users", {"name": "John Doe"})
    print(result)

    # Update document
    update = {"age": 31}
    repository.update_one("users", {"name": "John Doe"}, update)

    # Delete document
    repository.delete_one("users", {"name": "John Doe"})
