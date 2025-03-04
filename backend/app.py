from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
from werkzeug.utils import secure_filename
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import PyPDF2
from docx import Document
from model.anonymizer import anonymize_text
from model.summarizer import summarize_text
from utils.file_handler import convert_to_input_format
from utils.auth import verify_google_token
from pymongo import MongoClient
from bson.objectid import ObjectId

app = Flask(__name__)
CORS(app, origins=["https://yourusername.github.io"])  # Update with GitHub Pages URL

# Configuration
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
GOOGLE_CLIENT_ID = "your-google-client-id"  # Get from Google Cloud Console
MONGO_URI = "mongodb://localhost:27017"  # Update for Render (e.g., MongoDB Atlas)
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['legaldocs_ai']
users_collection = db['users']
summaries_collection = db['summaries']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/auth/google', methods=['POST'])
def google_auth():
    token = request.json.get('token')
    try:
        user_info = verify_google_token(token, GOOGLE_CLIENT_ID)
        user_id = user_info['sub']
        user = users_collection.find_one({'google_id': user_id})
        if not user:
            users_collection.insert_one({
                'google_id': user_id,
                'email': user_info['email'],
                'name': user_info['name']
            })
        return jsonify({'user_id': user_id, 'email': user_info['email']})
    except ValueError as e:
        return jsonify({'error': str(e)}), 401

@app.route('/api/process', methods=['POST'])
def process_document():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Authentication required'}), 401
    token = auth_header.split(' ')[1]
    user_info = verify_google_token(token, GOOGLE_CLIENT_ID)
    user_id = user_info['sub']

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    anonymize = request.form.get('anonymize', 'true').lower() == 'true'
    summarize = request.form.get('summarize', 'true').lower() == 'true'

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        text = extract_text(filepath, filename.rsplit('.', 1)[1].lower())
        result = {'original': text}
        if anonymize:
            result['anonymized'] = anonymize_text(text)
            anon_file = convert_to_input_format(result['anonymized'], filename.rsplit('.', 1)[1].lower())
        if summarize:
            result['summary'] = summarize_text(text)

        # Store in MongoDB
        doc_id = summaries_collection.insert_one({
            'user_id': user_id,
            'original': text,
            'summary': result.get('summary', ''),
            'anonymized': result.get('anonymized', ''),
            'timestamp': os.path.getctime(filepath)
        }).inserted_id

        os.remove(filepath)
        response = jsonify({'summary': result.get('summary', ''), 'doc_id': str(doc_id)})
        if anonymize:
            response.headers['Content-Disposition'] = f'attachment; filename=anon_{filename}'
            return send_file(anon_file, as_attachment=True, download_name=f'anon_{filename}')
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Authentication required'}), 401
    token = auth_header.split(' ')[1]
    user_info = verify_google_token(token, GOOGLE_CLIENT_ID)
    user_id = user_info['sub']

    history = list(summaries_collection.find({'user_id': user_id}, {'_id': 1, 'summary': 1, 'timestamp': 1}))
    return jsonify([{'id': str(h['_id']), 'summary': h['summary'], 'timestamp': h['timestamp']} for h in history])

def extract_text(filepath, file_type):
    if file_type == 'pdf':
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif file_type == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_type == 'docx':
        doc = Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])
    return "Unsupported file type"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)