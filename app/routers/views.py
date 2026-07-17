from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..storage import get_all_documents, get_document

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    docs = get_all_documents()
    return templates.TemplateResponse(request=request, name="index.html", context={"documents": docs})

@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse(request=request, name="upload.html", context={})

@router.get("/chat/{doc_id}", response_class=HTMLResponse)
async def chat_page(request: Request, doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return templates.TemplateResponse(request=request, name="chat.html", context={"document": doc})
