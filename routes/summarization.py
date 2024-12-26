from enum import Enum
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from text_summarizer import summarizer
from utils.file_converter import pdf_to_text
from langchain_core.documents import Document

summarization_router = APIRouter()

class SummarizationType(str, Enum):
    stuff = "stuff"
    map_reduce = "map_reduce"
    refine_chain = "refine"

@summarization_router.post("/summarize")
async def summarize(
    summarization_type: SummarizationType,
    text: Optional[str] = None,
    file: Optional[UploadFile] = None
):
    
    if text and file:
        raise HTTPException(status_code=400, detail="Please provide either text or file, not both")
    
    elif file:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        text = await pdf_to_text(file)
        summary = summarizer(text, summarization_type)
        return {"summary": summary}
    elif text:
        text = [Document(page_content=text)]
        summary = summarizer(text, summarization_type)
        return {"summary": summary}
    else:
        raise HTTPException(status_code=400, detail="Either text or file must be provided")