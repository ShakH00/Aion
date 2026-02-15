from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, jsonify
from pathlib import Path
import users
from users import user
from pymongo import MongoClient
import os
import re
import secrets
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
access_links_c = db['access_links']
UPLOAD_DIR = Path(app.root_path) / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = set(['.pdf', '.png', '.jpg', '.jpeg', '.txt'])
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 #10MB limit


def wants_json_response() -> bool:
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return True
    return request.accept_mimetypes["application/json"] >= request.accept_mimetypes["text/html"]


def has_doc_access(doc_id: str) -> bool:
    if 'email' in session:
        return True
    return session.get('access_doc_id') == doc_id


def create_access_link(doc_id: str, allow_download: bool = True) -> str:
    token = secrets.token_urlsafe(32)
    access_links_c.insert_one({
        "token": token,
        "doc_id": doc_id,
        "created_at": datetime.utcnow(),
        "allow_download": allow_download,
        "expires_at": None,
    })
    return token


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
                return jsonify(success=True, redirect=url_for('home'))
            return redirect(url_for('home'))
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
                return jsonify(success=True, redirect=url_for('home'))
            return redirect(url_for('home'))
    
    return send_from_directory(INDEX_DIR, "register.html")


@app.route('/access', methods=['GET', 'POST'])
def access_link():
    if request.method == 'GET':
        return send_from_directory(INDEX_DIR, "access.html")

    token = request.form.get('token')
    if not token:
        if wants_json_response():
            return jsonify(success=False, message='Please enter an access link.'), 400
        return redirect(url_for('access_link', error='Please enter an access link.'))

    record = access_links_c.find_one({"token": token})
    if not record:
        if wants_json_response():
            return jsonify(success=False, message='Invalid or expired access link.'), 404
        return redirect(url_for('access_link', error='Invalid or expired access link.'))

    doc_id = record.get("doc_id")
    if not doc_id:
        if wants_json_response():
            return jsonify(success=False, message='Invalid access link.'), 404
        return redirect(url_for('access_link', error='Invalid access link.'))

    session['access_token'] = token
    session['access_doc_id'] = doc_id
    if wants_json_response():
        return jsonify(success=True, redirect=url_for('viewdoc', doc_id=doc_id))
    return redirect(url_for('viewdoc', doc_id=doc_id))


@app.route('/access/<token>')
def access_link_direct(token):
    record = access_links_c.find_one({"token": token})
    if not record:
        return redirect(url_for('access_link', error='Invalid or expired access link.'))

    doc_id = record.get("doc_id")
    if not doc_id:
        return redirect(url_for('access_link', error='Invalid access link.'))

    session['access_token'] = token
    session['access_doc_id'] = doc_id
    return redirect(url_for('viewdoc', doc_id=doc_id))


@app.route('/access/create/<doc_id>')
def create_access_link_route(doc_id):
    if 'email' not in session:
        return redirect(url_for('login'))
    token = create_access_link(doc_id, allow_download=True)
    link = url_for('access_link_direct', token=token, _external=True)
    if wants_json_response():
        return jsonify(success=True, link=link)
    return link

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
    if not has_doc_access(doc_id):
        return redirect(url_for('login'))
    try:
        doc = docs_c.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return "Document not found", 404
    except:
        return "Invalid ID:", 400
    return render_template("reader.html", doc = doc)

@app.route('/files/<filename>')
def get_file(filename):
    doc = docs_c.find_one({"filename": filename})
    if not doc:
        return "File not found", 404
    if not has_doc_access(str(doc.get("_id"))):
        return redirect(url_for('login'))
    return send_from_directory(UPLOAD_DIR, filename)

#logout method
@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('index'))

if __name__ == "__main__":
	app.run(debug=True)
