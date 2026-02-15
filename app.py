from flask import Flask, send_from_directory
from pathlib import Path


app = Flask(__name__)
INDEX_DIR = Path(app.root_path) / "index"


@app.route("/")
def index():
    return send_from_directory(INDEX_DIR, "index.html")


if __name__ == "__main__":
	app.run(debug=True)
