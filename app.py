import bcrypt
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, jsonify, send_file, Response
from pathlib import Path
import users
from users import user
from pymongo import MongoClient
from gridfs import GridFS
import os
import re
import secrets
from datetime import datetime, timezone
from bson import ObjectId
import time
import base64

import fitz
from PIL import Image
import pytesseract
import io
import textwrap
import time
from scanner_ocr import scan_document_with_gemini



from google import genai
from google.genai import types
import requests

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

client = MongoClient(os.getenv("DB_KEY"))
db = client['aionDB']
fs = GridFS(db)  # Initialize GridFS for file storage
col = db['users']
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-change-me")
INDEX_DIR = Path(app.root_path) / "index"

#pdf stuff
docs_c = db['documents']
access_links_c = db['access_links']
# UPLOAD_DIR is no longer used - files stored in MongoDB GridFS
ALLOWED_EXTENSIONS = set(['.pdf', '.png', '.jpg', '.jpeg', '.txt'])
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 #10MB limit

# Create database indexes for performance (lazy creation, non-blocking)
def create_indexes():
    try:
        # Index on uploaded_by for faster user document queries
        docs_c.create_index("uploaded_by", background=True)
        # Index on is_public for faster public library queries
        docs_c.create_index("is_public", background=True)
        # Index on created_at for faster sorting
        docs_c.create_index("created_at", background=True)
        # Index on filename for faster file lookups
        docs_c.create_index("filename", background=True)
        # Compound index for common query patterns
        docs_c.create_index([("uploaded_by", 1), ("created_at", -1)], background=True)
        docs_c.create_index([("is_public", 1), ("created_at", -1)], background=True)
        print("Database indexes created successfully")
    except Exception as e:
        print(f"Note: Indexes may already exist: {e}")

# Create indexes in background after first request
@app.before_request
def setup_indexes():
    if not hasattr(app, 'indexes_created'):
        app.indexes_created = True
        from threading import Thread
        Thread(target=create_indexes, daemon=True).start()


def wants_json_response() -> bool:
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return True
    return request.accept_mimetypes["application/json"] >= request.accept_mimetypes["text/html"]


def has_doc_access(doc_id: str) -> bool:
    # Check if user is logged in
    if 'email' in session:
        return True
    # Check if document is public (only fetch is_public field)
    try:
        doc = docs_c.find_one(
            {"_id": ObjectId(doc_id)},
            {"is_public": 1}  # Only fetch is_public field
        )
        if doc and doc.get('is_public', False):
            return True
    except:
        pass
    # Check if user has access link
    return session.get('access_doc_id') == doc_id


def create_access_link(doc_id: str, allow_download: bool = True) -> str:
    token = secrets.token_urlsafe(32)
    access_links_c.insert_one({
        "token": token,
        "doc_id": doc_id,
        "created_at": datetime.now(timezone.utc),
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
            return redirect(url_for('login', error='Invalid email or password.'))

        pwd_hash = user['password']
        if not verify_hash(pwd_hash, pwd):
            if wants_json_response():
                return jsonify(success=False, message='Invalid email or password.'), 401
            return redirect(url_for('login', error='Invalid email or password.'))

        session['email'] = username  # Log the user in after registration
        if wants_json_response():
            return jsonify(success=True, redirect=url_for('home'))
        return redirect(url_for('home'))

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

        user = col.find_one({"email": email})
        if user:
            if wants_json_response():
                return jsonify(success=False, message='That email is already registered.'), 409
            return redirect(url_for('register', error='That email is already registered.'))

        hashed_pwd = hash_(pwd)
        dic = {"email":email, "password":hashed_pwd, "name": first, "lastname": last}
        user = col.insert_one(dic)
        session['email'] = email  # Log the user in after registration
        if wants_json_response():
            return jsonify(success=True, redirect=url_for('home'))
        return redirect(url_for('home'))

    
    return send_from_directory(INDEX_DIR, "register.html")


@app.route('/access', methods=['GET', 'POST'])
def access_link():
    if request.method == 'GET':
        return send_from_directory(INDEX_DIR, "access.html")

    # Handle both 'token' (from dedicated access page) and 'code' (from login page)
    token = request.form.get('token') or request.form.get('code')
    if not token:
        if wants_json_response():
            return jsonify(success=False, message='Please enter an access code.'), 400
        return redirect(url_for('access_link', error='Please enter an access code.'))

    record = access_links_c.find_one({"token": token})
    if not record:
        if wants_json_response():
            return jsonify(success=False, message='Invalid or expired access code.'), 404
        return redirect(url_for('access_link', error='Invalid or expired access code.'))

    doc_id = record.get("doc_id")
    if not doc_id:
        if wants_json_response():
            return jsonify(success=False, message='Invalid access code.'), 404
        return redirect(url_for('access_link', error='Invalid access code.'))

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
    if wants_json_response():
        return jsonify(success=True, token=token)
    return token

#helper for brf
def format_brf(text, cols=40, rows=25):
    lines = []
    for para in (text or "").replace("\r", "").split("\n"):
        if not para.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width=cols))

    out = []
    for i, line in enumerate(lines):
        out.append(line[:cols].ljust(cols))
        if (i + 1) % rows == 0:
            out.append("\f")
    return "\n".join(out).strip()

# helpers for pdf
def secure_filename_basic(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    return name[:200] if name else "file"

def extract_text_from_pdf(file_content, lang="eng"):
    text_chunks = []
    doc = fitz.open(stream=file_content, filetype="pdf")

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


def extract_text_from_pics(file_content, lang="eng"):
    """Extract text from images with OCR preprocessing for better accuracy"""
    import cv2
    import numpy as np
    
    img = Image.open(file_content).convert("RGB")
    
    # Convert PIL image to OpenCV format for preprocessing
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Upscale image for better OCR (common improvement for small/low-res images)
    height, width = cv_img.shape[:2]
    if width < 800:
        scale = 800 / width
        cv_img = cv2.resize(cv_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)
    
    # Apply threshold for better text detection
    _, thresh = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Auto-rotation detection (skew correction)
    # Find contours and correct rotation if needed
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] or [np.array([[0, 0], [0, 1], [1, 1], [1, 0]])])[2]
    
    if -45 < angle < 45 and angle != 0:
        h, w = thresh.shape
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        thresh = cv2.warpAffine(thresh, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Convert back to PIL for OCR
    processed_img = Image.fromarray(thresh)
    
    return pytesseract.image_to_string(processed_img, lang=lang)


#upload pdf
@app.route('/uploadpdf', methods=['GET', 'POST'])
def uploadpdf():
    if 'email' not in session:
        return redirect(url_for('login'))

    if request.method == 'GET':
        return send_from_directory(INDEX_DIR, "upload.html")

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
    is_public = request.form.get('is_public') == 'true'

    # Create doc_id first
    doc_id = ObjectId()

    # Extract text for search
    file_content = file.read()
    file.seek(0)  # Reset file pointer
    
    extracted_text = ""
    try:
        if ext == '.pdf':
            extracted_text = extract_text_from_pdf(io.BytesIO(file_content), language)
        elif ext in ['.png', '.jpg', '.jpeg']:
            extracted_text = extract_text_from_pics(io.BytesIO(file_content), language)
        elif ext == '.txt':
            extracted_text = file_content.decode("utf-8", errors="ignore")
    except Exception as e:
        extracted_text = f"[Extraction failed: {str(e)}]"

    # Store file in GridFS
    mime_type = file.mimetype or "application/octet-stream"
    file_id = fs.put(file_content, filename=f"{doc_id}{ext}", content_type=mime_type)

    #mongo save
    # Get user's full name
    user_doc = col.find_one({"email": session['email']})
    user_name = f"{user_doc.get('name', '')} {user_doc.get('lastname', '')}".strip() if user_doc else session['email']
    
    docs_c.insert_one({
        "_id": doc_id,\
        "title": title,
        "authors": authors,
        "tags": tags.split(','),
        "date": date,
        "language" : language,
        "file_id" : file_id,  # GridFS file ID
        "filename" : f"{doc_id}{ext}",
        "original_name" : original_name,
        "uploaded_by" : session['email'],
        "uploaded_by_name" : user_name,
        "created_at" : datetime.now(timezone.utc),
        "text" : extracted_text,
        "is_public" : is_public,
        "edit_history" : [],
        "mime_type": mime_type,
    })

    return redirect(url_for('edit_record', doc_id=str(doc_id)))


#upload throuogh camera
@app.route("/api/device/create-token")
def create_device_token():
    if 'email' not in session:
        return jsonify(success=False, message="Not authenticated."), 401

    token = secrets.token_urlsafe(32)

    access_links_c.insert_one({
        "token": token,
        "type": "device_upload",
        "owner": session['email'],
        "created_at": datetime.now(timezone.utc),
    })

    return jsonify(success=True, token=token)

@app.route("/api/camera/upload", methods=["POST"])
def camera_upload():
    token = request.headers.get("X-Device-Token")
    if not token:
        return jsonify(success=False, message="Missing device token."), 401

    rec = access_links_c.find_one({"token": token, "type": "device_upload"})
    if not rec:
        return jsonify(success=False, message="Invalid device token."), 403

    owner_email = rec.get("owner")
    if not owner_email:
        return jsonify(success=False, message="Invalid device token."), 403

    if 'file' not in request.files:
        return jsonify(success=False, message="No file provided."), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(success=False, message="Empty filename."), 400

    original_name = secure_filename_basic(file.filename)
    ext = Path(original_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        return jsonify(success=False, message="Unsupported file type."), 400

    title = request.form.get('title', original_name)
    authors = request.form.get('authors', 'Raspberry Pi')
    tags = request.form.get('tags', 'pi-scan,camera')
    date = request.form.get('date', time.strftime("%Y-%m-%d"))
    language = request.form.get('language', 'eng')
    is_public = request.form.get('is_public') == 'true'

    doc_id = ObjectId()

    file_content = file.read()

    extracted_text = ""
    try:
        if ext == '.pdf':
            extracted_text = extract_text_from_pdf(io.BytesIO(file_content), language)
        elif ext in ['.png', '.jpg', '.jpeg']:
            extracted_text = extract_text_from_pics(io.BytesIO(file_content), language)
        elif ext == '.txt':
            extracted_text = file_content.decode("utf-8", errors="ignore")
    except Exception as e:
        extracted_text = f"[Extraction failed: {str(e)}]"

    mime_type = file.mimetype or "application/octet-stream"
    file_id = fs.put(file_content, filename=f"{doc_id}{ext}", content_type=mime_type)

    user_doc = col.find_one({"email": owner_email})
    user_name = f"{user_doc.get('name', '')} {user_doc.get('lastname', '')}".strip() if user_doc else owner_email

    docs_c.insert_one({
        "_id": doc_id,
        "title": title,
        "authors": authors,
        "tags": tags.split(','),
        "date": date,
        "language": language,
        "file_id": file_id,
        "filename": f"{doc_id}{ext}",
        "original_name": original_name,
        "uploaded_by": owner_email,
        "uploaded_by_name": user_name,
        "created_at": datetime.now(timezone.utc),
        "text": extracted_text,
        "is_public": is_public,
        "edit_history": [],
        "mime_type": mime_type,
    })

    return jsonify(success=True, doc_id=str(doc_id))


@app.route("/api/analyze-file", methods=["POST"])
def api_analyze_file():
    if 'email' not in session:
        return jsonify(success=False, message="Not authenticated."), 401

    if 'file' not in request.files:
        return jsonify(success=False, message="No file provided."), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(success=False, message="Empty filename."), 400

    original_name = secure_filename_basic(file.filename)
    ext = Path(original_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        return jsonify(success=False, message="Unsupported file type."), 400

    file_content = file.read()

    extracted_text = ""
    try:
        if ext == '.pdf':
            extracted_text = extract_text_from_pdf(io.BytesIO(file_content), "eng")
        elif ext in ['.png', '.jpg', '.jpeg']:
            extracted_text = extract_text_from_pics(io.BytesIO(file_content), "eng")
        elif ext == '.txt':
            extracted_text = file_content.decode("utf-8", errors="ignore")
    except Exception as e:
        extracted_text = ""

    # SUPER basic “AI-ish” defaults (you can improve later)
    guess_title = Path(original_name).stem
    guess_authors = ""
    guess_tags = ["scan"] if ext in [".png", ".jpg", ".jpeg"] else ["document"]

    return jsonify(
        success=True,
        title=guess_title,
        authors=guess_authors,
        tags=guess_tags,
        preview=(extracted_text[:800] if extracted_text else "")
    )


@app.route('/mylibrary')
def library():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    return send_from_directory(INDEX_DIR, "library.html")


@app.route('/api/library')
def api_library():
    if 'email' not in session:
        return jsonify(success=False, message='Not authenticated.'), 401
    
    # Exclude large text field for performance
    user_docs = list(docs_c.find(
        {"uploaded_by": session['email']},
        {"text": 0}  # Exclude text field
    ).sort("created_at", -1).limit(100))  # Limit to 100 most recent documents
    
    records = []
    for doc in user_docs:
        records.append({
            "_id": str(doc["_id"]),
            "title": doc.get("title", "Untitled"),
            "authors": doc.get("authors", ""),
            "tags": doc.get("tags", []),
            "date": doc.get("date", ""),
            "is_public": doc.get("is_public", True),
            "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else ""
        })
    
    return jsonify(success=True, records=records)


@app.route('/public-library')
def public_library():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    return send_from_directory(INDEX_DIR, "public-library.html")


@app.route('/api/public-library')
def api_public_library():
    if 'email' not in session:
        return jsonify(success=False, message='Not authenticated.'), 401
    
    # Exclude large text field for performance
    public_docs = list(docs_c.find(
        {"is_public": True},
        {"text": 0}  # Exclude text field
    ).sort("created_at", -1).limit(100))  # Limit to 100 most recent documents
    
    records = []
    for doc in public_docs:
        records.append({
            "_id": str(doc["_id"]),
            "title": doc.get("title", "Untitled"),
            "authors": doc.get("authors", ""),
            "tags": doc.get("tags", []),
            "date": doc.get("date", ""),
            "uploaded_by": doc.get("uploaded_by", ""),
            "uploaded_by_name": doc.get("uploaded_by_name", ""),
            "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else ""
        })
    
    return jsonify(success=True, records=records)


@app.route('/api/file/<doc_id>')
def api_get_file(doc_id):
    if 'email' not in session:
        return jsonify(success=False, message='Not authenticated.'), 401
    
    try:
        doc = docs_c.find_one({"_id": ObjectId(doc_id), "uploaded_by": session['email']})
        if not doc:
            return jsonify(success=False, message='Record not found.'), 404
        
        doc_data = {
            "_id": str(doc["_id"]),
            "title": doc.get("title", ""),
            "authors": doc.get("authors", ""),
            "tags": doc.get("tags", []),
            "date": doc.get("date", ""),
            "language": doc.get("language", "en"),
            "filename": doc.get("filename", ""),
            "is_public": doc.get("is_public", True),
            "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else ""
        }
        
        return jsonify(success=True, doc=doc_data)
    except:
        return jsonify(success=False, message='Invalid ID.'), 400


@app.route('/file/<doc_id>')
def edit_record(doc_id):
    if 'email' not in session:
        return redirect(url_for('login'))
    
    try:
        doc = docs_c.find_one({"_id": ObjectId(doc_id), "uploaded_by": session['email']})
        if not doc:
            return "Record not found or access denied", 404
    except:
        return "Invalid ID", 400
    
    return send_from_directory(INDEX_DIR, "edit.html")


@app.route('/file/<doc_id>/update', methods=['POST'])
def update_record(doc_id):
    if 'email' not in session:
        return redirect(url_for('login'))
    
    try:
        doc = docs_c.find_one({"_id": ObjectId(doc_id), "uploaded_by": session['email']})
        if not doc:
            if wants_json_response():
                return jsonify(success=False, message='Record not found.'), 404
            return "Record not found", 404
    except:
        if wants_json_response():
            return jsonify(success=False, message='Invalid ID.'), 400
        return "Invalid ID", 400
    
    title = request.form.get('title', doc.get('title'))
    authors = request.form.get('authors', doc.get('authors'))
    tags = request.form.get('tags', '')
    date = request.form.get('date', doc.get('date'))
    is_public = request.form.get('is_public') == 'true'
    
    # Track changes for edit history
    changes = {}
    if title != doc.get('title'):
        changes['title'] = {'old': doc.get('title'), 'new': title}
    if authors != doc.get('authors'):
        changes['authors'] = {'old': doc.get('authors'), 'new': authors}
    if tags and (tags.split(',') != doc.get('tags', [])):
        changes['tags'] = {'old': doc.get('tags', []), 'new': tags.split(',')}
    if date != doc.get('date'):
        changes['date'] = {'old': doc.get('date'), 'new': date}
    if is_public != doc.get('is_public'):
        changes['is_public'] = {'old': doc.get('is_public'), 'new': is_public}
    
    # Get user's full name for edit history
    user_doc = col.find_one({"email": session['email']})
    user_name = f"{user_doc.get('name', '')} {user_doc.get('lastname', '')}".strip() if user_doc else session['email']
    
    # Add to edit history if there are changes
    edit_record = None
    if changes:
        edit_record = {
            "edited_by": session['email'],
            "edited_by_name": user_name,
            "edited_at": datetime.now(timezone.utc),
            "changes": changes
        }
    
    update_data = {
        "title": title,
        "authors": authors,
        "tags": tags.split(',') if tags else doc.get('tags', []),
        "date": date,
        "is_public": is_public,
        "updated_at": datetime.now(timezone.utc)
    }
    
    # Add edit record to history if there are changes
    if edit_record:
        update_data['edit_history'] = doc.get('edit_history', []) + [edit_record]
    
    docs_c.update_one(
        {"_id": ObjectId(doc_id)},
        {"$set": update_data}
    )
    
    if wants_json_response():
        return jsonify(success=True, message='Record updated.')
    return redirect(url_for('edit_record', doc_id=doc_id))


@app.route('/file/<doc_id>/generate-link')
def generate_access_link(doc_id):
    if 'email' not in session:
        return redirect(url_for('login'))
    
    try:
        doc = docs_c.find_one({"_id": ObjectId(doc_id), "uploaded_by": session['email']})
        if not doc:
            if wants_json_response():
                return jsonify(success=False, message='Record not found.'), 404
            return "Record not found", 404
    except:
        if wants_json_response():
            return jsonify(success=False, message='Invalid ID.'), 400
        return "Invalid ID", 400
    
    token = create_access_link(doc_id, allow_download=True)
    if wants_json_response():
        return jsonify(success=True, token=token)
    return jsonify(success=True, token=token)

@app.route('/doc/<doc_id>')
def viewdoc(doc_id):
    if not has_doc_access(doc_id):
        return redirect(url_for('login'))
    try:
        doc = docs_c.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return "Document not found", 404
    except:
        return "Invalid ID", 400
    return send_from_directory(INDEX_DIR, "reader.html")


@app.route('/api/doc/<doc_id>')
def api_get_document(doc_id):
    if not has_doc_access(doc_id):
        return jsonify(success=False, message='Access denied.'), 403
    
    try:
        doc = docs_c.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify(success=False, message='Document not found.'), 404
        
        # Check if user can edit (owner only)
        can_edit = 'email' in session and doc.get('uploaded_by') == session['email']
        
        doc_data = {
            "_id": str(doc["_id"]),
            "title": doc.get("title", "Untitled"),
            "authors": doc.get("authors", ""),
            "tags": doc.get("tags", []),
            "date": doc.get("date", ""),
            "language": doc.get("language", "en"),
            "filename": doc.get("filename", ""),
            "text": doc.get("text", ""),
            "can_edit": can_edit,
            "uploaded_by_name": doc.get("uploaded_by_name", doc.get("uploaded_by", "Unknown")),
            "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else "",
            "edit_history": doc.get("edit_history", []),
            "mime_type": doc.get("mime_type", "application/octet-stream"),
            
        }
        
        return jsonify(success=True, doc=doc_data)
    except:
        return jsonify(success=False, message='Invalid document ID.'), 400


@app.route('/files/<filename>')
def get_file(filename):
    doc = docs_c.find_one({"filename": filename})
    if not doc:
        return "File not found", 404
    if not has_doc_access(str(doc.get("_id"))):
        return redirect(url_for('login'))
    
    # Retrieve file from GridFS
    file_id = doc.get("file_id")
    if not file_id:
        return "File not found in storage", 404
    
    try:
        file_data = fs.get(file_id)
        mime_type = doc.get("mime_type") or getattr(file_data, "content_type", None) or "application/octet-stream"
        
        disp = "inline" if (mime_type.startswith("image/") or mime_type == "application/pdf") else "attachment"
        return file_data.read(), 200, {
            "Content-Type": mime_type,
            "Content-Disposition": f'{disp}; filename="{doc.get("original_name")}"'
            }
    except:
        return "Error retrieving file", 500
    
@app.route("/api/pi/scan", methods=["POST"])
def api_pi_scan():
    if 'email' not in session:
        return jsonify(success=False, message="Not logged in"), 401

    try:
        image_bytes, extracted_text, saved_path = scan_document_with_gemini(timeout=90, save_image=True)
    except TimeoutError:
        return jsonify(success=False, message="No button press detected."), 408
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

    doc_id = ObjectId()
    mime_type = "image/jpeg"
    file_id = fs.put(image_bytes, filename=f"{doc_id}.jpg", content_type=mime_type)

    user_doc = col.find_one({"email": session['email']})
    user_name = f"{user_doc.get('name','')} {user_doc.get('lastname','')}".strip() if user_doc else session['email']

    docs_c.insert_one({
        "_id": doc_id,
        "title": f"Pi Scan {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "authors": "Raspberry Pi",
        "tags": ["pi-scan", "camera"],
        "date": time.strftime("%Y-%m-%d"),
        "language": "eng",
        "file_id": file_id,
        "filename": f"{doc_id}.jpg",
        "original_name": "pi_scan.jpg",
        "uploaded_by": session['email'],
        "uploaded_by_name": user_name,
        "created_at": datetime.now(timezone.utc),
        "text": extracted_text,
        "is_public": True,
        "edit_history": [],
        "mime_type": mime_type,
    })

    return jsonify(success=True, doc_id=str(doc_id), preview=extracted_text[:400])





#brf FILE STUFF
@app.route("/api/doc/<doc_id>/export/brf")
def export_brf(doc_id):
    if not has_doc_access(doc_id):
        return jsonify(success=False, message="Access denied."), 403
    try:
        doc = docs_c.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return {"error": "Not found"}, 404
    except:
        return {"error": "Invalid ID"}, 400
    
    brf_text = format_brf(doc.get("text", ""))

    buf = io.BytesIO(brf_text.encode("utf-8"))
    buf.seek(0)

    return send_file(
        buf,
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"{doc_id}.brf"
    )




#logout method
@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('index'))

# Get edit history for a document
@app.route('/api/doc/<doc_id>/edit-history')
def get_edit_history(doc_id):
    if 'email' not in session:
        return jsonify(success=False, message='Not authenticated.'), 401
    
    if not has_doc_access(doc_id):
        return jsonify(success=False, message='Access denied.'), 403
    
    try:
        doc = docs_c.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify(success=False, message='Document not found.'), 404
        
        edit_history = doc.get('edit_history', [])
        
        # Format timestamps for readability
        formatted_history = []
        for edit in edit_history:
            formatted_edit = {
                'edited_by_name': edit.get('edited_by_name', edit.get('edited_by')),
                'edited_at': edit.get('edited_at').isoformat() if edit.get('edited_at') else 'Unknown',
                'changes': edit.get('changes', {})
            }
            formatted_history.append(formatted_edit)
        
        return jsonify(success=True, edit_history=list(reversed(formatted_history)))
    except Exception as e:
        print(f"Error getting edit history: {str(e)}")
        return jsonify(success=False, message=f'Error: {str(e)}'), 500


# Download file from edit history
@app.route('/api/doc/<doc_id>/file-version/<filename>', methods=['GET'])
def download_file_version(doc_id, filename):
    if 'email' not in session:
        return jsonify(success=False, message='Not authenticated.'), 401
    
    try:
        doc = docs_c.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify(success=False, message='Document not found.'), 404
        
        # Check if user can access this document
        if doc.get('uploaded_by') != session['email'] and not doc.get('is_public', False):
            return jsonify(success=False, message='Access denied.'), 403
        
        # Get file from GridFS
        files = fs.find({"filename": filename})
        file_list = list(files)
        
        if not file_list:
            return jsonify(success=False, message='File not found.'), 404
        
        file_obj = file_list[0]
        file_content = file_obj.read()
        
        return send_file(
            io.BytesIO(file_content),
            download_name=filename,
            as_attachment=True
        )
    except Exception as e:
        print(f"Error downloading file version: {str(e)}")
        return jsonify(success=False, message=f'Error: {str(e)}'), 500


# Upload new file for document
@app.route('/api/doc/<doc_id>/upload-file', methods=['POST'])
def upload_new_file(doc_id):
    if 'email' not in session:
        return jsonify(success=False, message='Not authenticated.'), 401
    
    try:
        doc = docs_c.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify(success=False, message='Document not found.'), 404
        
        # Check if user can edit this document
        if doc.get('uploaded_by') != session['email']:
            return jsonify(success=False, message='Access denied.'), 403
        
        # Get the uploaded file
        if 'file' not in request.files:
            return jsonify(success=False, message='No file provided.'), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify(success=False, message='No file selected.'), 400
        
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return jsonify(success=False, message=f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'), 400
        
        # Read file content
        file_content = file.read()
        if len(file_content) == 0:
            return jsonify(success=False, message='File is empty.'), 400
        
        # Get language from document or use default
        language = doc.get('language', 'eng')
        
        # Extract text from the new file
        extracted_text = ""
        try:
            if file_ext == '.pdf':
                extracted_text = extract_text_from_pdf(io.BytesIO(file_content), language)
            elif file_ext in ['.png', '.jpg', '.jpeg']:
                extracted_text = extract_text_from_pics(io.BytesIO(file_content), language)
            elif file_ext == '.txt':
                extracted_text = file_content.decode("utf-8", errors="ignore")
        except Exception as e:
            return jsonify(success=False, message=f'Failed to extract text: {str(e)}'), 400
        
        # Keep old file in GridFS (don't delete for archival purposes)
        old_filename = doc.get('filename')
        old_file_id = doc.get('file_id')
        
        # Store new file in GridFS
        new_filename = f"{doc_id}{file_ext}"
        mime_type = file.mimetype or "application/octet-stream"
        file_id = fs.put(file_content, filename=new_filename, content_type=mime_type)
        
        # Get user's full name for edit history
        user_doc = col.find_one({"email": session['email']})
        user_name = f"{user_doc.get('name', '')} {user_doc.get('lastname', '')}".strip() if user_doc else session['email']
        
        # Create edit history record with file version tracking
        edit_record = {
            "edited_by": session['email'],
            "edited_by_name": user_name,
            "edited_at": datetime.now(timezone.utc),
            "changes": {
                "file": {
                    "old": old_filename or 'None',
                    "new": new_filename,
                    "old_file_id": str(old_file_id) if old_file_id else None
                }
            }
        }
        
        # Update document with new file info and edit history
        docs_c.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {
                "filename": new_filename,
                "file_id": str(file_id),
                "full_text": extracted_text,
                "text": extracted_text,
                "updated_at": datetime.now(timezone.utc),
                "edit_history": doc.get('edit_history', []) + [edit_record]
            }}
        )
        
        return jsonify(success=True, message='File uploaded and text extracted successfully.')
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        return jsonify(success=False, message=f'Error: {str(e)}'), 500



@app.route('/api/translate', methods=['POST'])
def translate_text():
    if 'email' not in session:
        return jsonify(success=False, message='Not authenticated.'), 401
    
    if not gemini_client:
        return jsonify(success=False, message='AI translation not available. Please try again later.'), 503
    
    data = request.get_json()
    text = data.get('text', '').strip()
    target_language = data.get('target_language', 'es')
    
    if not text or len(text) < 1:
        return jsonify(success=False, message='Text cannot be empty.'), 400
    
    # Language name mapping
    lang_names = {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
        'zh': 'Chinese', 'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi',
        'th': 'Thai', 'vi': 'Vietnamese', 'nl': 'Dutch', 'pl': 'Polish',
        'tr': 'Turkish', 'id': 'Indonesian', 'tl': 'Tagalog'
    }
    
    target_lang = target_language
    target_lang_name = lang_names.get(target_lang, target_lang)
    
    try:
        print(f"Translating {len(text)} chars to {target_lang_name} using Gemini AI...")
        
        # Use Gemini to translate
        prompt = f"""Translate the following text to {target_lang_name}. 
Only return the translated text, nothing else. Do not add explanations or comments.

Text to translate:
{text}"""
        
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        if not response or not response.text:
            print(f"  Gemini returned empty response")
            return jsonify(success=False, message='Translation service returned empty response'), 503
        
        translated_text = response.text.strip()
        
        if not translated_text or translated_text == text:
            print(f"  Gemini translation failed or returned original text")
            return jsonify(success=False, message='Translation service failed'), 503
        
        print(f"  Success! Translated to {len(translated_text)} chars")
        
        return jsonify(
            success=True,
            translated_text=translated_text,
            source_language='auto',
            target_language=target_lang
        )
        
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return jsonify(success=False, message=f'Translation failed: {str(e)}'), 500

# Configure Gemini AI
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if gemini_api_key:
    try:
        gemini_client = genai.Client()
    except Exception as e:
        print(f"Failed to create Gemini client: {e}")
        gemini_client = None
else:
    gemini_client = None
    print("Warning: GOOGLE_API_KEY not set. AI search will not work.")

# Comprehensive accessibility document analysis  
@app.route('/api/doc/<doc_id>/accessibility-analysis', methods=['GET'])
def accessibility_analysis(doc_id):
    """Extract full document content including image descriptions using Gemini vision"""
    # Note: No session check required - accessibility is a read operation for document viewers
    
    if not gemini_client:
        return jsonify(success=False, message='AI analysis not available.'), 503
    
    try:
        # Try to parse doc_id as ObjectId
        try:
            doc_obj_id = ObjectId(doc_id)
        except:
            return jsonify(success=False, message='Invalid document ID.'), 400
        
        # Get document from database
        doc = db.documents.find_one({'_id': doc_obj_id})
        if not doc:
            return jsonify(success=False, message='Document not found.'), 404
        
        # Check if we already have cached analysis
        if doc.get('accessibility_analysis'):
            print(f"✓ Returning cached accessibility analysis for {doc.get('title', 'document')}")
            return jsonify(success=True, data=doc['accessibility_analysis'])
        
        # Get the actual file
        if not doc.get('file_id'):
            return jsonify(success=False, message='No file content.'), 400
        
        try:
            file_content = fs.get(ObjectId(doc['file_id'])).read()
        except:
            return jsonify(success=False, message='Cannot read file.'), 500
        
        result = {
            'title': doc.get('title'),
            'authors': doc.get('authors'),
            'language': doc.get('language'),
            'text_content': '',
            'images': [],
            'accessibility_summary': ''
        }
        
        file_extension = (doc.get('filename', '')).lower().split('.')[-1] if doc.get('filename') else ''
        
        # Extract text based on file type
        if file_extension in ['pdf']:
            result['text_content'] = extract_text_from_pdf(io.BytesIO(file_content), doc.get('language', 'eng'))
            
            # Extract images from PDF and store their xref for later alt-text generation
            print(f"Extracting image metadata from PDF...")
            try:
                pdf_doc = fitz.open(stream=io.BytesIO(file_content), filetype="pdf")
                image_count = 0
                for page_num, page in enumerate(pdf_doc):
                    images = page.get_images()
                    for img_idx, img in enumerate(images):
                        image_count += 1
                    result.setdefault('image_count', 0)
                    result['image_count'] = max(result.get('image_count', 0), image_count)
                print(f"Found {image_count} images in PDF")
            except Exception as e:
                print(f"Could not extract image metadata: {str(e)}")
        
        elif file_extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
            result['text_content'] = extract_text_from_pics(io.BytesIO(file_content), doc.get('language', 'eng'))
            print(f"Image file detected - will handle alt-text generation in reader")
        
        elif file_extension in ['txt']:
            try:
                with io.TextIOWrapper(io.BytesIO(file_content), encoding='utf-8') as text_file:
                    result['text_content'] = text_file.read()
            except:
                result['text_content'] = extract_text_from_pics(io.BytesIO(file_content), doc.get('language', 'eng'))
        
        else:
            result['text_content'] = extract_text_from_pdf(io.BytesIO(file_content), doc.get('language', 'eng'))
        
        # Generate accessibility summary using Gemini (HTML format)
        print(f"Generating accessibility summary...")
        summary_prompt = f"""Create a comprehensive accessibility summary of this document. Format your response as PURE HTML.

IMPORTANT: Do NOT wrap your response in ```html or any markdown code fences. Return ONLY the raw HTML tags.

Title: {result['title']}
Authors: {result['authors']}

Main Text Content (first 2000 chars):
{result['text_content'][:2000] if result['text_content'] else 'No text content'}

Provide as ONLY raw HTML tags (use <h3>, <p>, <ul>, <li> only):
- A brief overview of the document
- Key topics covered
- Recommendations for accessibility
- Summary for screen reader users

Return ONLY the HTML, no markdown, no code fences, no explanations."""
        
        summary_response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=summary_prompt
        )
        
        if summary_response and summary_response.text:
            summary_html = summary_response.text.strip()
            summary_html = re.sub(r"^```(?:html)?\s*", "", summary_html, flags=re.IGNORECASE)
            summary_html = re.sub(r"```\s*$", "", summary_html)
            result['accessibility_summary'] = summary_html.strip()
        
        # Cache the analysis in the database
        print(f"Caching accessibility analysis in database...")
        db.documents.update_one(
            {'_id': doc_obj_id},
            {'$set': {'accessibility_analysis': result}}
        )
        
        print(f"Accessibility analysis complete!")
        
        return jsonify(
            success=True,
            data=result
        )
    
    except Exception as e:
        print(f"Accessibility analysis error: {str(e)}")
        return jsonify(success=False, message=f'Analysis failed: {str(e)}'), 500

# Configure Google Cloud Translation
# (No longer needed - using Gemini AI)
print("[OK] Translation using Gemini AI")
print("[OK] Comprehensive accessibility analysis using Gemini vision")

@app.route('/api/doc/<doc_id>/image-alt-text', methods=['POST'])
def generate_image_alt_text(doc_id):
    """Generate alt-text for images in a document using Gemini vision"""
    if not gemini_client:
        return jsonify(success=False, message='AI analysis not available.'), 503
    
    try:
        data = request.get_json() or {}
        image_index = int(data.get('image_index', 0))
        
        # Try to parse doc_id as ObjectId
        try:
            doc_obj_id = ObjectId(doc_id)
        except:
            return jsonify(success=False, message='Invalid document ID.'), 400
        
        # Get document from database
        doc = db.documents.find_one({'_id': doc_obj_id})
        if not doc:
            return jsonify(success=False, message='Document not found.'), 404
        
        # Get the file
        if not doc.get('file_id'):
            return jsonify(success=False, message='No file content.'), 400
        
        try:
            file_content = fs.get(ObjectId(doc['file_id'])).read()
        except:
            return jsonify(success=False, message='Cannot read file.'), 500
        
        file_extension = (doc.get('filename', '')).lower().split('.')[-1] if doc.get('filename') else ''
        
        # Return cached alt-text if available
        cached_alts = doc.get('pdf_image_alt_texts', []) or []
        for cached in cached_alts:
            if cached.get('index') == image_index and cached.get('alt_text'):
                return jsonify(success=True, alt_text=cached['alt_text'])

        # Extract image based on file type
        image_data = None
        image_page = 1
        
        if file_extension == 'pdf':
            # Extract image from PDF
            try:
                pdf_doc = fitz.open(stream=io.BytesIO(file_content), filetype="pdf")
                
                # Find the image
                img_count = 0
                for page_num, page in enumerate(pdf_doc):
                    images = page.get_images()
                    for img_idx, img in enumerate(images):
                        if img_count == image_index:
                            # Found the image we want
                            xref = img[0]
                            pix = fitz.Pixmap(pdf_doc, xref)
                            
                            # Convert to PIL and then JPEG
                            if pix.n - pix.alpha < 4:
                                pil_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                            else:
                                pil_image = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
                            
                            pil_image = pil_image.convert("RGB")
                            pil_image.thumbnail((768, 768), Image.LANCZOS)
                            jpeg_buffer = io.BytesIO()
                            pil_image.save(jpeg_buffer, format="JPEG", quality=70)
                            image_data = jpeg_buffer.getvalue()
                            image_page = page_num + 1
                            break
                        img_count += 1
                    if image_data:
                        break
            except Exception as e:
                print(f"Error extracting image from PDF: {str(e)}")
                return jsonify(success=False, message='Could not extract image from PDF.'), 500
        
        elif file_extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
            # Use the entire file as the image
            if image_index == 0:
                try:
                    pil_image = Image.open(io.BytesIO(file_content))
                    jpeg_buffer = io.BytesIO()
                    if pil_image.mode in ('RGBA', 'LA', 'P'):
                        pil_image = pil_image.convert('RGB')
                    pil_image.save(jpeg_buffer, format="JPEG", quality=85)
                    image_data = jpeg_buffer.getvalue()
                except Exception as e:
                    print(f"Error processing image: {str(e)}")
                    return jsonify(success=False, message='Could not process image.'), 500
            else:
                return jsonify(success=False, message='Image index out of range.'), 400
        else:
            return jsonify(success=False, message='File type does not contain images.'), 400
        
        if not image_data:
            return jsonify(success=False, message='Could not extract image.'), 400
        
        # Generate alt-text using Gemini vision
        print(f"Generating alt-text for image {image_index}...")
        
        try:
            try:
                image_part = types.Part.from_data(
                    mime_type="image/jpeg",
                    data=image_data
                )
            except AttributeError:
                image_part = types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=image_data
                )
            
            alt_text_response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    "Write one natural-sounding alt-text sentence. Describe what is shown and any readable text. No lists, no options, no labels, no quotes. Keep under 150 characters.",
                    image_part
                ]
            )
            
            if alt_text_response and alt_text_response.text:
                alt_text = alt_text_response.text.strip()
                print(f"✓ Generated alt-text: {alt_text[:80]}...")
                try:
                    db.documents.update_one(
                        {'_id': doc_obj_id},
                        {'$push': {'pdf_image_alt_texts': {'index': image_index, 'page': image_page, 'alt_text': alt_text}}}
                    )
                except Exception:
                    pass

                return jsonify(success=True, alt_text=alt_text)
            else:
                return jsonify(success=False, message='Failed to generate alt-text.'), 500
        
        except Exception as e:
            print(f"Error generating alt-text: {str(e)}")
            return jsonify(success=False, message=f'Alt-text generation failed: {str(e)}'), 500
    
    except Exception as e:
        print(f"Image alt-text error: {str(e)}")
        return jsonify(success=False, message=f'Error: {str(e)}'), 500


@app.route('/api/generate-mp3', methods=['POST'])
def generate_mp3():
    """Generate audio file from text using pyttsx3"""
    try:
        import pyttsx3
        import tempfile
        import sys
        
        data = request.get_json() or {}
        text = data.get('text', '')
        language = data.get('language', 'en-US')
        rate = data.get('rate', 1.0)
        
        print(f"\n=== Audio Generation ===")
        print(f"Text length: {len(text)} chars")
        print(f"Language: {language}")
        print(f"Rate: {rate}")
        
        if not text:
            return jsonify(success=False, message='No text provided.'), 400
        
        try:
            # Initialize pyttsx3 engine
            print(f"Initializing pyttsx3 engine...")
            if sys.platform.startswith("win"):
                engine = pyttsx3.init('sapi5')
            else:
                engine = pyttsx3.init()
            
            # Get available voices
            voices = engine.getProperty('voices')
            print(f"Available voices: {len(voices)}")
            for i, voice in enumerate(voices):
                print(f"  Voice {i}: {voice.name} - {voice.languages}")
            
            # Try to find matching voice
            voice_found = False
            for voice in voices:
                try:
                    if voice.languages and len(voice.languages) > 0:
                        if language.lower() in str(voice.languages[0]).lower():
                            engine.setProperty('voice', voice.id)
                            voice_found = True
                            print(f"Found matching voice: {voice.name}")
                            break
                except:
                    continue
            
            if not voice_found and voices:
                engine.setProperty('voice', voices[0].id)
                print(f"Using default voice: {voices[0].name}")
            
            # Set rate (pyttsx3 typically uses 50-300, default ~200)
            rate_value = max(50, min(300, int(150 * float(rate))))
            engine.setProperty('rate', rate_value)
            print(f"Rate set to: {rate_value}")
            
            # Set volume
            engine.setProperty('volume', 1.0)
            
            # Create temp file
            print(f"Creating temporary file...")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_audio_path = tmp_file.name
            
            print(f"Temp file: {temp_audio_path}")
            
            try:
                # Generate audio
                print(f"Saving to file...")
                engine.save_to_file(text, temp_audio_path)
                
                print(f"Running engine...")
                engine.runAndWait()
                
                # Check if file was created
                if not os.path.exists(temp_audio_path):
                    print(f"ERROR: Temp file was not created!")
                    return jsonify(success=False, message='Audio file generation failed - temp file not created.'), 500
                
                file_size = os.path.getsize(temp_audio_path)
                print(f"Audio file created: {file_size} bytes")
                
                # Read file
                with open(temp_audio_path, 'rb') as f:
                    audio_data = f.read()
                
                print(f"Read {len(audio_data)} bytes from file")
                print(f"=== Audio Generation Complete ===")
                
                return Response(
                    audio_data,
                    mimetype='audio/wav',
                    headers={'Content-Disposition': 'attachment; filename=document_speech.wav'}
                )
            
            finally:
                # Clean up
                try:
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                        print(f"Temp file cleaned up")
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temp file: {cleanup_error}")
                try:
                    engine.stop()
                except Exception:
                    pass
        
        except Exception as engine_error:
            print(f"pyttsx3 Engine Error: {str(engine_error)}")
            import traceback
            traceback.print_exc()
            return jsonify(success=False, message=f'Audio generation failed: {str(engine_error)}'), 500
    
    except ImportError as ie:
        print(f"Import Error: {str(ie)}")
        return jsonify(success=False, message='pyttsx3 not installed.'), 503
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify(success=False, message=f'Error: {str(e)}'), 500


@app.route('/api/doc/<doc_id>/pdf-images', methods=['GET'])
def extract_pdf_images(doc_id):
    """Extract all images from PDF with alt-text"""
    try:
        # Try to parse doc_id as ObjectId
        try:
            doc_obj_id = ObjectId(doc_id)
        except:
            return jsonify(success=False, message='Invalid document ID.'), 400
        
        # Get document from database
        doc = db.documents.find_one({'_id': doc_obj_id})
        if not doc:
            return jsonify(success=False, message='Document not found.'), 404
        
        # Get the file
        if not doc.get('file_id'):
            return jsonify(success=False, message='No file content.'), 400
        
        try:
            file_content = fs.get(ObjectId(doc['file_id'])).read()
        except:
            return jsonify(success=False, message='Cannot read file.'), 500
        
        file_extension = (doc.get('filename', '')).lower().split('.')[-1] if doc.get('filename') else ''
        
        if file_extension != 'pdf':
            return jsonify(success=False, message='Not a PDF file.'), 400
        
        include_data = str(request.args.get('include_data', 'false')).lower() == 'true'
        images = []

        cached_alts = doc.get('pdf_image_alt_texts', []) or []
        cached_count = doc.get('pdf_image_count')
        
        try:
            pdf_doc = fitz.open(stream=io.BytesIO(file_content), filetype="pdf")
            
            image_index = 0
            for page_num, page in enumerate(pdf_doc):
                page_images = page.get_images()
                
                for img_idx, img in enumerate(page_images):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_doc, xref)
                        
                        if pix.n - pix.alpha < 4:
                            pil_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                        else:
                            pil_image = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)

                        pil_image = pil_image.convert("RGB")
                        thumbnail_image = pil_image.copy()
                        thumbnail_image.thumbnail((768, 768), Image.LANCZOS)

                        buffer = io.BytesIO()
                        if include_data:
                            pil_image.save(buffer, format="PNG")
                        else:
                            thumbnail_image.save(buffer, format="JPEG", quality=70)

                        cached_alt = next((item for item in cached_alts if item.get('index') == image_index), None)
                        alt_text = cached_alt.get('alt_text') if cached_alt else None

                        if not alt_text:
                            try:
                                try:
                                    image_part = types.Part.from_data(
                                        mime_type="image/jpeg",
                                        data=buffer.getvalue()
                                    )
                                except AttributeError:
                                    image_part = types.Part.from_bytes(
                                        mime_type="image/jpeg",
                                        data=buffer.getvalue()
                                    )

                                alt_response = gemini_client.models.generate_content(
                                    model="gemini-2.0-flash",
                                    contents=[
                                        "Write one natural-sounding alt-text sentence. Describe what is shown and any readable text. No lists, no options, no labels, no quotes. Keep under 150 characters.",
                                        image_part
                                    ]
                                )

                                if alt_response and alt_response.text:
                                    alt_text = alt_response.text.strip()
                                    cached_alts.append({'index': image_index, 'page': page_num + 1, 'alt_text': alt_text})
                            except Exception as e:
                                print(f"Error generating alt-text for image {image_index}: {str(e)}")
                                alt_text = None
                        
                        image_payload = {
                            'index': image_index,
                            'page': page_num + 1,
                            'width': pix.width,
                            'height': pix.height,
                            'alt_text': alt_text or ""
                        }

                        if include_data:
                            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            image_payload['data'] = f'data:image/png;base64,{base64_image}'

                        images.append(image_payload)
                        
                        image_index += 1
                        print(f"Extracted image {image_index} from page {page_num + 1}")
                        
                    except Exception as e:
                        print(f"Error extracting image {img_idx} from page {page_num}: {str(e)}")
                        continue
        
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return jsonify(success=False, message=f'Error processing PDF: {str(e)}'), 500
        
        try:
            if cached_alts:
                db.documents.update_one(
                    {'_id': doc_obj_id},
                    {'$set': {'pdf_image_alt_texts': cached_alts, 'pdf_image_count': image_index}}
                )
        except Exception:
            pass

        return jsonify(
            success=True,
            images=images,
            total_pages=len(pdf_doc) if 'pdf_doc' in locals() else 0,
            cached=True
        )
    
    except Exception as e:
        print(f"PDF extraction error: {str(e)}")
        return jsonify(success=False, message=f'Error: {str(e)}'), 500


@app.route('/api/doc/<doc_id>/text-by-page', methods=['GET'])
def extract_text_by_page(doc_id):
    """Return text split by page for inline placement of image descriptions"""
    try:
        try:
            doc_obj_id = ObjectId(doc_id)
        except Exception:
            return jsonify(success=False, message='Invalid document ID.'), 400

        doc = db.documents.find_one({'_id': doc_obj_id})
        if not doc:
            return jsonify(success=False, message='Document not found.'), 404

        if not doc.get('file_id'):
            return jsonify(success=False, message='No file content.'), 400

        try:
            file_content = fs.get(ObjectId(doc['file_id'])).read()
        except Exception:
            return jsonify(success=False, message='Cannot read file.'), 500

        file_extension = (doc.get('filename', '')).lower().split('.')[-1] if doc.get('filename') else ''

        if file_extension != 'pdf':
            if file_extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                text_content = extract_text_from_pics(io.BytesIO(file_content), doc.get('language', 'eng'))
            elif file_extension == 'txt':
                try:
                    with io.TextIOWrapper(io.BytesIO(file_content), encoding='utf-8') as text_file:
                        text_content = text_file.read()
                except Exception:
                    text_content = extract_text_from_pics(io.BytesIO(file_content), doc.get('language', 'eng'))
            else:
                text_content = extract_text_from_pdf(io.BytesIO(file_content), doc.get('language', 'eng'))

            return jsonify(success=True, pages=[{'page': 1, 'text': text_content or ''}])

        pages = []
        pdf_doc = fitz.open(stream=io.BytesIO(file_content), filetype="pdf")
        direct_pages = []

        for page in pdf_doc:
            direct_pages.append(page.get_text("text") or "")

        direct_text = "\n".join(direct_pages).strip()

        if len(direct_text) > 200:
            for idx, text in enumerate(direct_pages, start=1):
                pages.append({'page': idx, 'text': text})
        else:
            for idx, page in enumerate(pdf_doc, start=1):
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img, lang=doc.get('language', 'eng'))
                pages.append({'page': idx, 'text': ocr_text})

        return jsonify(success=True, pages=pages)

    except Exception as e:
        print(f"Text-by-page error: {str(e)}")
        return jsonify(success=False, message=f'Error: {str(e)}'), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
