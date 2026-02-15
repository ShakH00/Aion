import bcrypt
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
col = db['users']
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


@app.route("/home")
def home():
    if 'email' not in session:
        return redirect(url_for('login'))
    return send_from_directory(INDEX_DIR, "home.html")


#helper method to get the username (first and last name) of an account
def getUserName():
    all_users = users.get_all_users()
    userName = ""
    for user in all_users:
        if user['email'] == session['email']:
            userName += f"{user['name']} {user['lastname']}"
    return userName


"""
Helper function to hash passwords for security purposes
"""
def hash_(pwd):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd.encode('utf-8'), salt)
    return hashed.decode('utf-8')

"""
Helper function to verify a stored password against one provided by the user
"""
def verify_hash(stored_pwd, provided_pwd):

    stored_pwd = stored_pwd.encode("utf-8")

    return bcrypt.checkpw(provided_pwd.encode("utf-8"), stored_pwd)

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

        user =  col.find_one({"email": username})
        if user is None:
            if wants_json_response():
                return jsonify(success=False, message='Invalid email or password.'), 401

        pwd_hash = user['password']
        if not verify_hash(pwd_hash, pwd):
            if wants_json_response():
                return jsonify(success=False, message='Invalid email or password.'), 401

        session['email'] = username  # Log the user in after registration
        if wants_json_response():
            return jsonify(success=True, redirect=url_for('index'))
        return redirect(url_for('index'))

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
        user = col.find({"email": email})
        if user:
            if wants_json_response():
                return jsonify(success=False, message='That email is already registered.'), 409
            return redirect(url_for('register', error='That email is already registered.'))

        hashed_pwd = hash_(pwd)
        dic = {"email":email, "password":hashed_pwd, "name": first, "lastname": last}
        user = col.insert_one(dic)
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
    if 'email' not in session:
        return redirect(url_for('login'))

    if request.method == 'GET':
        return render_template("upload.html")

    #POST methods
    if 'file' not in request.files:
        return "No file provided", 400

    file = request.files['file']
    if file.filename == '':
        return "Empty Filename", 400

    original_name = secure_filename_basic(file.filename)
    ext = Path(original_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        return "Unsuported file type", 400

    #metadata from form
    title = request.form.get('title', original_name)
    authors = request.form.get('authors', ' ')
    tags = request.form.get('tags', ' ')
    date = request.form.get('date', '')
    language = request.form.get('language', 'eng')

    #db id
    doc_id = ObjectId()
    stored_name = f"{doc_id}{ext}"
    save_path = UPLOAD_DIR / stored_name
    file.save(save_path)

    extracted_text = ""
    try:
        if ext == '.pdf':
            extracted_text = extract_text_from_pdf(save_path, language)
        elif ext in ['.png', '.jpg', 'jpeg']:
            extracted_text = extract_text_from_pics(save_path, language)
        elif ext == '.txt':
            extracted_text = save_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        extracted_text = f"[Extraction failed: {str(e)}]"

    #mongo save
    docs_c.insert_one({
        "_id": doc_id,\
        "title": title,
        "authors": authors,
        "tags": tags.split(','),
        "date": date,
        "language" : language,
        "filename" : stored_name,
        "original_name" : original_name,
        "uploaded_by" : session['email'],
        "created_at" : datetime.utcnow(),
        "text" : extracted_text,
    })

    return redirect(url_for('viewdoc', doc_id = str(doc_id)))

@app.route('/doc/<doc_id>')
def viewdoc(doc_id):
    try:
        doc = docs_c.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return "Document not found", 404
    except:
        return "Invalid ID:", 400
    return render_template("reader.html", doc = doc)

@app.route('/files/<filename>')
def get_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

#logout method
@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('index'))

if __name__ == "__main__":
	app.run(debug=True)
