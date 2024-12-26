import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from fastapi import UploadFile

async def pdf_to_text(file: UploadFile):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        # Write the uploaded file content to the temporary file
        content = await file.read()
        tmp_file.write(content)
        tmp_file.flush()
        
        try:
            pdf_reader = PyPDFLoader(tmp_file.name)
            documents = pdf_reader.load()
            return documents
        finally:
            # Make sure we clean up by attempting to remove the file
            try:
                os.unlink(tmp_file.name)
            except PermissionError:
                # If we can't delete now, try to delete on next garbage collection
                import atexit
                atexit.register(lambda: os.unlink(tmp_file.name) if os.path.exists(tmp_file.name) else None)