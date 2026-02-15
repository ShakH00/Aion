from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
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
INDEX_DIR = Path(app.root_path) / "index"

#pdf stuff
docs_c = db['documents']
UPLOAD_DIR = Path(app.root_path) / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = set(['.pdf', '.png', '.jpg', '.jpeg', '.txt'])
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 #10MB limit


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
        email, password = users.get_user_credentials(username)
        #verify password
        if user.verify_password(password, pwd):  # Simulate successful login
            session['email'] = username
            return redirect(url_for('home'))
        else:
            return send_from_directory(INDEX_DIR, "login.html")
    return send_from_directory(INDEX_DIR, "login.html")


#register account
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        pwd = request.form.get('password')
        first = request.form.get('first')
        last = request.form.get('last')

        if email is None or pwd is None or first is None or last is None:
            return send_from_directory(INDEX_DIR, "register.html")

        exists = False
        all_users = users.get_all_users()
        #for loop and if statement used to make sure the email isn't already in use
        for usr in all_users:
            if usr['email'] == email:
                exists = True
        if exists == True:
            return send_from_directory(INDEX_DIR, "register.html")
        else:
            new_person = user(first, last, email, pwd)
            new_person.insert_doc()

            # Simulate user registration
            session['email'] = email  # Log the user in after registration
            return redirect(url_for('login'))
    
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
    return redirect(url_for('home'))

if __name__ == "__main__":
	app.run(debug=True)
