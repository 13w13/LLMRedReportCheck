import gradio as gr
import json
from datetime import datetime
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

# Backend functions (from your original code)
def read_docx(file) -> str:
    try:
        content = file.read()
        doc = Document(io.BytesIO(content))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Failed to read DOCX file: {file.name} - {e}")
        raise gr.Error(f"DOCX file read failed: {str(e)}")

def read_excel(file) -> pd.DataFrame:
    try:
        content = file.read()
        return pd.read_excel(io.BytesIO(content))
    except Exception as e:
        logger.error(f"Failed to read Excel file: {file.name} - {e}")
        raise gr.Error(f"Excel file read failed: {str(e)}")

# Include other backend functions: chunk_text, hybrid_search, generate_llm_response, generate_validation_questions

def process_files(narrative_file, indicators_file, ns_data_file, financial_overview_file, bilateral_support_file):
    if all([narrative_file, indicators_file, ns_data_file, financial_overview_file, bilateral_support_file]):
        try:
            narrative_text = read_docx(narrative_file)
            narrative_chunks = chunk_text(narrative_text)
            indicators_data = read_excel(indicators_file)
            ns_data = read_excel(ns_data_file)
            financial_overview = read_excel(financial_overview_file)
            bilateral_support = read_excel(bilateral_support_file)

            validation_questions = generate_validation_questions(
                narrative_chunks, indicators_data, ns_data, financial_overview, bilateral_support
            )
            
            formatted_questions = ""
            questions = validation_questions.split('\n\n')
            for i, q in enumerate(questions, 1):
                formatted_questions += f"Question {i}:\n"
                lines = q.split('\n')
                if len(lines) >= 7:
                    formatted_questions += f"Country: {lines[0].split(': ')[1]}\n"
                    formatted_questions += f"Created: {lines[1].split(': ')[1]}\n"
                    formatted_questions += f"Status: {lines[2].split(': ')[1]}\n"
                    formatted_questions += f"Section: {lines[3].split(': ')[1]}\n"
                    formatted_questions += f"Indicator: {lines[4].split(': ')[1]}\n"
                    formatted_questions += f"Reported value: {lines[5].split(': ')[1]}\n"
                    formatted_questions += f"Question: {lines[6].split(': ')[1]}\n\n"
            
            return formatted_questions
        
        except Exception as e:
            return f"An error occurred: {str(e)}"
    else:
        return "Please upload all required files."

def export_json(formatted_questions):
    questions = formatted_questions.split("Question")[1:]  # Skip the first empty split
    export_data = []
    for q in questions:
        lines = q.strip().split('\n')
        if len(lines) >= 7:
            export_data.append({
                "country": lines[0].split(': ')[1],
                "created": lines[1].split(': ')[1],
                "status": lines[2].split(': ')[1],
                "section": lines[3].split(': ')[1],
                "indicator": lines[4].split(': ')[1],
                "reported_value": lines[5].split(': ')[1],
                "question": lines[6].split(': ')[1]
            })
    
    return json.dumps(export_data, indent=2)

# Define the Gradio interface
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🔍 LLM-RedReportCheck")
    gr.Markdown("Upload narrative and data files to generate validation questions.")
    
    with gr.Row():
        with gr.Column():
            narrative_file = gr.File(label="Upload Narrative Report (DOCX)", file_types=[".docx"])
            indicators_file = gr.File(label="Upload Indicators (XLSX)", file_types=[".xlsx"])
            ns_data_file = gr.File(label="Upload National Society Data (XLSX)", file_types=[".xlsx"])
        with gr.Column():
            financial_overview_file = gr.File(label="Upload Financial Overview (XLSX)", file_types=[".xlsx"])
            bilateral_support_file = gr.File(label="Upload Bilateral Support (XLSX)", file_types=[".xlsx"])
    
    questions_output = gr.Textbox(label="Validation Questions", lines=10)
    
    generate_btn = gr.Button("Generate Validation Questions")
    generate_btn.click(fn=process_files, inputs=[narrative_file, indicators_file, ns_data_file, financial_overview_file, bilateral_support_file], outputs=questions_output)
    
    export_btn = gr.Button("Export Questions as JSON")
    json_output = gr.JSON(label="Exported JSON")
    export_btn.click(fn=export_json, inputs=questions_output, outputs=json_output)
    
    gr.Markdown("---")
    gr.Markdown(
        "LLM-RedReportCheck is an AI-powered tool designed to validate "
        "humanitarian data reports. It uses Large Language Models to cross-check "
        "narrative reports against numerical data, identifying inconsistencies "
        "and generating validation questions."
    )
    gr.Markdown("Developed with ❤️ by Your Team | © 2023 LLM-RedReportCheck")

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()