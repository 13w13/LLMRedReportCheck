import gradio as gr
import json
from datetime import datetime
import io
import logging
from typing import List
import pandas as pd
import numpy as np
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
import os
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get Hugging Face token from environment variable
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")

# Llama model configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Load indicators list
INDICATORS_LIST = pd.read_csv('indicators_list.csv')

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

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

def create_db(splits):
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma.from_texts(
        texts=splits,
        embedding=embedding,
    )
    return vectordb

def initialize_llm():
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_NAME,
        temperature=0.7,
        max_new_tokens=500,
        top_k=3,
    )
    return llm

def generate_validation_questions(narrative_chunks: List[str], indicators: pd.DataFrame, ns_data: pd.DataFrame, financial_overview: pd.DataFrame, bilateral_support: pd.DataFrame):
    llm = initialize_llm()
    vectordb = create_db(narrative_chunks)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    
    all_questions = []
    
    for chunk in narrative_chunks:
        prompt = f"""
        You are an AI assistant specialized in validating Red Cross reports. Generate 1-2 detailed validation questions based on the following:

        Narrative Summary:
        {chunk}

        All Available Indicators:
        {INDICATORS_LIST.to_string()}

        For each question, provide:
        Country: Albania
        Created: {datetime.now().strftime('%Y-%m-%d')}
        Status: Open
        Section: [Relevant section from the data]
        Indicator: [Relevant indicator]
        Reported value: [Value from data or "Not Reported"]
        Question: [Your validation question]

        Focus on inconsistencies, gaps, and adherence to Red Cross reporting standards.
        """

        response = qa_chain({"question": prompt})
        all_questions.append(response['answer'])

    return "\n\n".join(all_questions)

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
            
            formatted_questions = format_questions(validation_questions)
            return formatted_questions
        
        except Exception as e:
            return f"An error occurred: {str(e)}"
    else:
        return "Please upload all required files."

def format_questions(validation_questions):
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

def export_json(formatted_questions):
    questions = formatted_questions.split("Question")[1:]
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

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()