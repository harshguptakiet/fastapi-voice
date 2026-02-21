from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.document_service import document_service
from app.schemas.document import DocumentIngestionResponse

router = APIRouter(prefix="/documents", tags=["documents"])



@router.post("/upload", response_model=DocumentIngestionResponse)
async def upload_document(file: UploadFile = File(...)):
    # Validate extension
    allowed_extensions = {".pdf", ".docx", ".pptx"}
    ext = f".{file.filename.split('.')[-1].lower()}"
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )

    content = await file.read()
    result = await document_service.save_and_extract(file.filename, content)
    
    return result

@router.get("/list")
async def list_documents():
    # Simple list of processed documents
    from pathlib import Path
    
    docs = []
    text_dir = Path("storage/documents/text")
    if text_dir.exists():
        for f in text_dir.glob("*.txt"):
            docs.append(f.stem)
    return {"documents": docs}
