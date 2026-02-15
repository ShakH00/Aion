from flask import Flask, send_from_directory, session
from pathlib import Path
import users
from users import user
from pymongo import MongoClient
import os

client = MongoClient(os.getenv("DB_KEY"))
db = client['aionDB']
user_c = db['users']
app = Flask(__name__)
INDEX_DIR = Path(app.root_path) / "index"


@app.route("/")
def index():
    return send_from_directory(INDEX_DIR, "index.html")


#helper method to get the username (first and last name) of an account
def getUserName():
    all_users = users.get_all_users()
    userName = ""
    for user in all_users:
        if user['email'] == session['email']:
            userName += f"{user['name']} {user['lastname']}"
    return userName

if __name__ == "__main__":
	app.run(debug=True)
