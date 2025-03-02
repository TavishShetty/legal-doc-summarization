from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from werkzeug.utils import secure_filename
import PyPDF2
from model.anonymizer import anonymize_text
from model.summarizer import summarize_text

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Sample documents (can move to files in samples/)
SAMPLES = {
    'invoice': """INVOICE #1234-5678\nDate: 15 January 2025\nBill To: John Smith\n123 Main Street, Mumbai\nPhone: 9123456789\nItems:\n1. Legal consultation - $500\nTotal: $500""",
    'agreement': """AGREEMENT\nDate: 10 January 2025\nBetween: ABC Corp, Delhi\nAnd: Jane Doe, Bangalore\nTerms: Confidentiality""",
    'bank': """BANK STATEMENT\nAccount Holder: Robert Johnson\nAccount: 1234-5678-9012\nPeriod: 01-31 Jan 2025\nTransactions:\n01/05/2025 | Deposit | $1500""",
    'medical': """MEDICAL CERTIFICATE\nPatient: Emma Wilson\nDOB: 03/12/1985\nDiagnosis: Flu\nDoctor: Dr. Sarah Miller"""
}

@app.route('/api/process', methods=['POST'])
def process_document():
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
        os.remove(filepath)

        result = {'original': text}
        if anonymize:
            result['anonymized'] = anonymize_text(text)
        if summarize:
            result['summary'] = summarize_text(text)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-sample', methods=['POST'])
def process_sample():
    data = request.get_json()
    if not data or 'sampleType' not in data:
        return jsonify({'error': 'Invalid request'}), 400

    sample_type = data['sampleType']
    if sample_type not in SAMPLES:
        return jsonify({'error': 'Sample not found'}), 404

    anonymize = data.get('anonymize', True)
    summarize = data.get('summarize', True)

    text = SAMPLES[sample_type]
    result = {'original': text}
    if anonymize:
        result['anonymized'] = anonymize_text(text)
    if summarize:
        result['summary'] = summarize_text(text)

    return jsonify(result)

def extract_text(filepath, file_type):
    if file_type == 'pdf':
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif file_type == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_type == 'docx':
        return "DOCX support requires python-docx library"  # Add python-docx if needed
    return "Unsupported file type"

@app.route('/')
def health_check():
    return "LegalDocs AI Backend is running", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)