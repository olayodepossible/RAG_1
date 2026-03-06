import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr

load_dotenv(override=True)

def load_documents(directory):
    f""""Load documents from {directory} directory."""
    print(f"Loading documents from {directory} directory...")

    #check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")
    
    loader = DirectoryLoader(directory, glob="**/*.txt",  loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into chunks with overlap."""
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)