import os
import shutil
from fastapi import UploadFile
from ..storage import get_doc_dir


def _safe_filename(filename: str) -> str:
    """Strip path separators and sanitize the filename to prevent path traversal."""
    # Take only the basename (handles both / and \ separators)
    name = os.path.basename(filename.replace("\\", "/"))
    # Remove null bytes and other dangerous characters
    name = name.replace("\x00", "").strip()
    return name if name else "document.pdf"


async def save_upload_file(upload_file: UploadFile, doc_id: str) -> str:
    doc_dir = get_doc_dir(doc_id)
    safe_name = _safe_filename(upload_file.filename or "document.pdf")
    file_path = os.path.join(doc_dir, safe_name)

    # Read in 1 MB chunks so large resumes/PDFs don't blow memory
    with open(file_path, "wb") as buffer:
        while content := await upload_file.read(1024 * 1024):
            buffer.write(content)

    return file_path
