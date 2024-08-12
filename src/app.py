import io
import logging
from typing import List
import pandas as pd
import numpy as np
from docx import Document
from rank_bm25 import BM25Okapi
from transformers import pipeline
import torch
import os
import gradio as gr

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get Hugging Face token from environment variable
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")

# Llama model configuration
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
llm = pipeline(
    "text-generation",
    model=MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN
)

# Load indicators list
INDICATORS_LIST = pd.read_csv('indicators_list.csv')

def read_docx(file) -> str:
    try:
        doc = Document(file.name)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Failed to read DOCX file: {file.name} - {e}")
        raise gr.Error(f"DOCX file read failed: {str(e)}")

def read_excel(file) -> pd.DataFrame:
    try:
        return pd.read_excel(file.name)
    except Exception as e:
        logger.error(f"Failed to read Excel file: {file.name} - {e}")
        raise gr.Error(f"Excel file read failed: {str(e)}")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def hybrid_search(query: str, chunks: List[str], top_k: int = 5) -> List[str]:
    tokenized_corpus = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.split())
    
    top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def generate_llm_response(prompt: str) -> str:
    response = llm(prompt, max_new_tokens=500, do_sample=True, temperature=0.7)
    return response[0]['generated_text']

def generate_validation_questions(narrative_chunks: List[str], indicators: pd.DataFrame, ns_data: pd.DataFrame, financial_overview: pd.DataFrame, bilateral_support: pd.DataFrame):
    all_questions = []
    
    for chunk in narrative_chunks:
        relevant_data = hybrid_search(chunk, indicators.to_string() + ns_data.to_string() + financial_overview.to_string() + bilateral_support.to_string())
        
        prompt = f"""
        You are an AI assistant specialized in validating Red Cross reports. Generate 1-2 detailed validation questions based on the following:

        Narrative Summary:
        {chunk}

        Relevant Data:
        {' '.join(relevant_data)}

        All Available Indicators:
        {INDICATORS_LIST.to_string()}

        For each question, provide:
        Country: Albania
        Created: [Current date]
        Status: Open
        Section: [Relevant section from the data]
        Indicator: [Relevant indicator]
        Reported value: [Value from data or "Not Reported"]
        Question: [Your validation question]

        Focus on inconsistencies, gaps, and adherence to Red Cross reporting standards.
        """

        response = generate_llm_response(prompt)
        all_questions.append(response)

    return "\n\n".join(all_questions)

def validate_data(narrative, indicators, ns_data, financial_overview, bilateral_support):
    try:
        narrative_text = read_docx(narrative)
        narrative_chunks = chunk_text(narrative_text)
        indicators_data = read_excel(indicators)
        ns_data = read_excel(ns_data)
        financial_overview = read_excel(financial_overview)
        bilateral_support = read_excel(bilateral_support)

        validation_questions = generate_validation_questions(
            narrative_chunks, indicators_data, ns_data, financial_overview, bilateral_support
        )

        return validation_questions
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return f"Validation process failed: {str(e)}"

# Gradio interface
iface = gr.Interface(
    fn=validate_data,
    inputs=[
        gr.File(label="Narrative (DOCX)"),
        gr.File(label="Indicators (Excel)"),
        gr.File(label="NS Data (Excel)"),
        gr.File(label="Financial Overview (Excel)"),
        gr.File(label="Bilateral Support (Excel)")
    ],
    outputs=gr.Textbox(label="Validation Questions"),
    title="LLM-RedReportCheck",
    description="Upload Red Cross report files to generate validation questions."
)

if __name__ == "__main__":
    iface.launch()