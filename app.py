from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, jsonify
from pathlib import Path
import users
from users import user
from pymongo import MongoClient
import os
import re
from datetime import datetime
from bson import ObjectId

import fitz
from PIL import Image
import pytesseract


client = MongoClient(os.getenv("DB_KEY"))
db = client['aionDB']
user_c = db['users']
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-change-me")
INDEX_DIR = Path(app.root_path) / "index"

#pdf stuff
docs_c = db['documents']
UPLOAD_DIR = Path(app.root_path) / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = set(['.pdf', '.png', '.jpg', '.jpeg', '.txt'])
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 #10MB limit


def wants_json_response() -> bool:
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return True
    return request.accept_mimetypes["application/json"] >= request.accept_mimetypes["text/html"]


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


#login method
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('email')
        pwd = request.form.get('password')
        if not username or not pwd:
            if wants_json_response():
                return jsonify(success=False, message='Please enter your email and password.'), 400
            return redirect(url_for('login', error='Please enter your email and password.'))
        email, password = users.get_user_credentials(username)
        if not email or not password:
            if wants_json_response():
                return jsonify(success=False, message='Invalid email or password.'), 401
            return redirect(url_for('login', error='Invalid email or password.'))
        #verify password
        if user.verify_password(password, pwd):  # Simulate successful login
            session['email'] = username
            if wants_json_response():
                return jsonify(success=True, redirect=url_for('index'))
            return redirect(url_for('index'))
        if wants_json_response():
            return jsonify(success=False, message='Invalid email or password.'), 401
        return redirect(url_for('login', error='Invalid email or password.'))
    return send_from_directory(INDEX_DIR, "login.html")


#register account
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        pwd = request.form.get('password')
        first = request.form.get('first')
        last = request.form.get('last')

        if not email or not pwd or not first or not last:
            if wants_json_response():
                return jsonify(success=False, message='Please fill out all fields.'), 400
            return redirect(url_for('register', error='Please fill out all fields.'))

        exists = False
        all_users = users.get_all_users()
        #for loop and if statement used to make sure the email isn't already in use
        for usr in all_users:
            if usr['email'] == email:
                exists = True
        if exists == True:
            if wants_json_response():
                return jsonify(success=False, message='That email is already registered.'), 409
            return redirect(url_for('register', error='That email is already registered.'))
        else:
            new_person = user(first, last, email, pwd)
            new_person.insert_doc()

            # Simulate user registration
            session['email'] = email  # Log the user in after registration
            if wants_json_response():
                return jsonify(success=True, redirect=url_for('index'))
            return redirect(url_for('index'))
    
    return send_from_directory(INDEX_DIR, "register.html")

# helpers for pdf
def secure_filename_basic(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    return name[:200] if name else "file"

def extract_text_from_pdf(pdf_path: Path, lang="eng"):
    text_chunks = []
    doc = fitz.open(str(pdf_path))

    for page in doc:
        t = page.get_text("text")
        if t:
            text_chunks.append(t)

    direct_text = "\n".join(text_chunks).strip()

    # use real text only
    if len(direct_text) > 200:
        return direct_text

    ocr_chunks = []
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_chunks.append(pytesseract.image_to_string(img, lang=lang))

    return "\n".join(ocr_chunks)

def extract_text_from_pics(img_path: Path, lang="eng"):
    img = image.open(img_path).convert("RGB")
    return pytesseract.image_to_string(img, lang=lang)


#upload pdf
@app.route('/uploadpdf', methods=['GET', 'POST'])
def uploadpdf():
    return render_template('uploadpdf.html') #temp

#logout method
@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('index'))

if __name__ == "__main__":
	app.run(debug=True)
