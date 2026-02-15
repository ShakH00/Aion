import os
import json

import bcrypt
from bson import json_util
from pymongo import MongoClient

client = MongoClient(os.getenv("DB_KEY"))
db = client['aionDB']
user_c = db['users']

class user:
    def __init__(self, name, lastname, email, password):
        self.name = name
        self.lastname = lastname
        self.email = email
        self.password = self.hash_password(password)


    # Convert user attributes to a dictionary
    def to_dict(self):
        return {
            "name": self.name,
            "lastname": self.lastname,
            "email": self.email,
            "password": self.password,
        }

    # Hash the password using bcrypt
    def hash_(self, password):
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed_password.decode('utf-8')

    def hash_password(self, password):
        return self.hash_(password)

    def insert_doc(self):
        # Use the dictionary representation for insertion
        insert_doc = user_c.insert_one(self.to_dict())
        print(f"Success, ID: {insert_doc.inserted_id}")

    def delete_doc(self):
        delete_doc = user_c.delete_one(self.to_dict())
        return delete_doc.deleted_count

    def find_doc(self):
        person = user_c.find_one(self.to_dict())
        return person

    def update_doc(self, identifier_doc, new_doc):
        # Ensure both parameters are dictionaries
        if isinstance(identifier_doc, user):
            identifier_doc = identifier_doc.to_dict()
        if isinstance(new_doc, user):
            new_doc = new_doc.to_dict()

        # Update the document in the database
        update_document = user_c.update_one(identifier_doc, {"$set": new_doc})
        return update_document.modified_count

        # Helper function to convert BSON documents to JSON format

    def bson_to_json(self, data):
        return json.dumps(json.loads(json_util.dumps(data)), indent=4)

    @staticmethod
    def verify_password(stored_password, provided_password):
        """Verify a stored password against one provided by the user."""
        if stored_password is None:
            return False
        if isinstance(stored_password, str):
            stored_bytes = stored_password.encode('utf-8')
        else:
            stored_bytes = stored_password
        return bcrypt.checkpw(provided_password.encode('utf-8'), stored_bytes)

def get_all_users():
    # Retrieve all user documents and convert to a list
    users = list(user_c.find())
    #all_users = get_all_users()
    return users


def get_user_credentials(email):
    # Use a dictionary as the filter
    curr = user_c.find_one({"email": email})

    # Check if a user document was found
    if curr:
        return curr["email"], curr["password"]
    else:
        return None, None  # Return None values if the user is not found