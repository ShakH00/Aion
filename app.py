import bcrypt
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, jsonify, send_file
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

import fitz
from PIL import Image
import pytesseract
import io
import textwrap

from google import genai
from google.genai import types

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
    img = Image.open(file_content).convert("RGB")
    return pytesseract.image_to_string(img, lang=lang)


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
        "mime_type": mime_type
    })

    return redirect(url_for('edit_record', doc_id=str(doc_id)))


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


@app.route('/api/ai-search', methods=['POST'])
def ai_search():
    if 'email' not in session:
        return jsonify(success=False, message='Not authenticated.'), 401
    
    if not gemini_client:
        return jsonify(success=False, message='AI search not configured. Please set GOOGLE_API_KEY.'), 503
    
    data = request.get_json()
    query = data.get('query', '').strip()
    search_type = data.get('type', 'private')  # 'private' or 'public'
    
    if not query:
        return jsonify(success=False, message='Query cannot be empty.'), 400
    
    try:
        # Get relevant documents based on search type
        if search_type == 'public':
            docs = list(docs_c.find(
                {"is_public": True},
                {"text": 1, "title": 1, "authors": 1, "date": 1, "tags": 1, "_id": 1, "created_at": 1}
            ).limit(50))
        else:
            docs = list(docs_c.find(
                {"uploaded_by": session['email']},
                {"text": 1, "title": 1, "authors": 1, "date": 1, "tags": 1, "_id": 1, "is_public": 1, "created_at": 1}
            ).limit(50))
        
        if not docs:
            return jsonify(success=True, results=[])
        
        # Build context for Gemini
        docs_context = []
        for i, doc in enumerate(docs):
            text_preview = (doc.get("text", "") or "")[:500]  # First 500 chars
            docs_context.append(f"""
Document {i+1}:
ID: {str(doc["_id"])}
Title: {doc.get("title", "Untitled")}
Authors: {doc.get("authors", "")}
Date: {doc.get("date", "")}
Tags: {", ".join(doc.get("tags", []))}
Content Preview: {text_preview}
""")
        
        # Create prompt for Gemini
        prompt = f"""You are helping search through a document library. The user's search query is: "{query}"

Here are the available documents:
{''.join(docs_context)}

Based on the user's query, analyze which documents are most relevant by considering:
1. Content similarity (semantic meaning in the text)
2. Author names matching or related to the query
3. Dates mentioned that match the query
4. Title and tags relevance

Return ONLY a JSON array of document IDs in order of relevance (most relevant first). Include only documents that are genuinely relevant to the query.
Format: ["id1", "id2", "id3"]

If no documents are relevant, return an empty array: []"""

        # Query Gemini
        response = gemini_client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        response_text = response.text.strip()
        
        # Extract JSON from response (handle markdown code blocks)
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        # Parse the response
        import json as json_lib
        relevant_ids = json_lib.loads(response_text)
        
        # Build result list with full document info
        results = []
        for doc_id in relevant_ids:
            doc = next((d for d in docs if str(d["_id"]) == doc_id), None)
            if doc:
                results.append({
                    "_id": str(doc["_id"]),
                    "title": doc.get("title", "Untitled"),
                    "authors": doc.get("authors", ""),
                    "tags": doc.get("tags", []),
                    "date": doc.get("date", ""),
                    "is_public": doc.get("is_public", True),
                    "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else ""
                })
        
        return jsonify(success=True, results=results, query=query)
    
    except Exception as e:
        print(f"AI search error: {str(e)}")
        return jsonify(success=False, message=f'AI search failed: {str(e)}'), 500


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

if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0')
