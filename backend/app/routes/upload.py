from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi_jwt_auth import AuthJWT
import fitz  # PyMuPDF
import docx
import os
from models.ml_model import summarize_text, anonymize_text
from config import Settings
import supabase

router = APIRouter()
settings = Settings()

# Initialize Supabase Client
supabase_client = supabase.create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# Function to extract text from different file types
def extract_text(file: UploadFile):
    text = ""
    try:
        if file.filename.endswith(".pdf"):
            doc = fitz.open(stream=file.file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text()
        elif file.filename.endswith(".docx"):
            doc = docx.Document(file.file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file.filename.endswith(".txt"):
            text = file.file.read().decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), anonymize: bool = True, summarize: bool = True, Authorize: AuthJWT = Depends()):
    Authorize.jwt_required()
    user_email = Authorize.get_jwt_subject()

    try:
        # Extract text from file
        text = extract_text(file)

        processed_text = text
        if anonymize:
            processed_text = anonymize_text(processed_text)
        if summarize:
            processed_text = summarize_text(processed_text)

        # Upload processed text to Supabase Storage
        file_path = f"documents/{user_email}/{file.filename}"
        supabase_client.storage.from_("documents").upload(file_path, processed_text.encode("utf-8"))

        return {"message": "File processed successfully", "download_url": f"{settings.SUPABASE_URL}/storage/v1/object/public/{file_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))