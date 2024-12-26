from langchain_groq import ChatGroq
from langchain.schema import(
    HumanMessage,
    SystemMessage,
    AIMessage
)
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from utils.text_splitter import split_text
from typing import List

def _stuff_summarizer(docs: List[Document], llm: ChatGroq):
    prompt=PromptTemplate(template=""" Write a concise and short summary of the following document,
    Document :{text}
    """
    )
    chain = load_summarize_chain(
        llm=llm,
        chain_type="stuff",
        prompt=prompt,
        verbose=True
    )
    return chain.run(docs)

def _map_reduce_summarizer(docs: List[Document], llm: ChatGroq):
    chunk_prompt = PromptTemplate(
        template=""" Write a concise and short summary of the following document,
        Document :{text}
        Summary :
        """
    )
    combine_prompt = PromptTemplate(
        template=""" Provide the final summary of the entire document with the following format,
        Format : Title, Genre, Summary.
        Documents :{text}
        """
    )
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=chunk_prompt,
        combine_prompt=combine_prompt,
        verbose=True
    )
    return chain.run(docs)

def _refine_chain_summarizer(docs: List[Document], llm: ChatGroq):
    prompt = PromptTemplate(
        template=""" Write a concise and short summary.
        """
    )
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        verbose=True,
    )
    return chain.run(docs)

def summarizer(docs: Document, summarization_type: str):
    llm = ChatGroq(model="llama3-8b-8192")
    if summarization_type == "stuff":
        return _stuff_summarizer(docs, llm)
    elif summarization_type == "map_reduce":
        split_docs = split_text(text=docs, chunk_size=500, chunk_overlap=100)
        return _map_reduce_summarizer(split_docs, llm)
    elif summarization_type == "refine":
        split_docs = split_text(text=docs, chunk_size=500, chunk_overlap=100)
        return _refine_chain_summarizer(split_docs, llm)

