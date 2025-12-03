from fastapi import APIRouter, UploadFile, File
import tempfile
from services.ocr_service import extract_text

router = APIRouter(prefix="/ocr", tags=["OCR"])

@router.post("/")
async def ocr_endpoint(file: UploadFile = File(...)):
    # نخزن الملف مؤقتًا
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(await file.read())
    temp.close()

    # نستدعي خدمة OCR
    extracted = extract_text(temp.name)

    return {
        "extracted_text": extracted
    }
