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
from langchain.schema import Document as LangchainDocument
import os
import torch
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time
import random
from pathlib import Path
import chromadb
from unidecode import unidecode
import re
from huggingface_hub import HfApi, HfHubHTTPError

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
INDICATORS_LIST = pd.read_csv('src/indicators_list.csv')

def read_docx(file) -> str:
    try:
        if hasattr(file, 'name'):
            doc = Document(file.name)
        elif isinstance(file, str):
            doc = Document(file)
        else:
            content = file.read() if hasattr(file, 'read') else file
            doc = Document(io.BytesIO(content))
        
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Failed to read DOCX file: {getattr(file, 'name', str(file))} - {e}")
        raise gr.Error(f"DOCX file read failed: {str(e)}")

def read_excel(file) -> pd.DataFrame:
    try:
        if hasattr(file, 'name'):
            return pd.read_excel(file.name)
        elif isinstance(file, str):
            return pd.read_excel(file)
        else:
            content = file.read() if hasattr(file, 'read') else file
            return pd.read_excel(io.BytesIO(content))
    except Exception as e:
        logger.error(f"Failed to read Excel file: {getattr(file, 'name', str(file))} - {e}")
        raise gr.Error(f"Excel file read failed: {str(e)}")

def chunk_text(text: str, chunk_size: int = 250, chunk_overlap: int = 50) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

def create_db(splits, collection_name):
    embedding = HuggingFaceEmbeddings()
    new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=new_client,
        collection_name=collection_name,
    )
    return vectordb

def initialize_llm(temperature=0.7, max_tokens=500, top_k=3):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=MODEL_NAME,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_k=top_k,
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise gr.Error(f"LLM initialization failed: {str(e)}")

def create_qa_chain(llm, vectordb):
    try:
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
        return qa_chain
    except Exception as e:
        logger.error(f"Failed to create QA chain: {str(e)}")
        raise gr.Error(f"QA chain creation failed: {str(e)}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(HfHubHTTPError))
def generate_question(qa_chain: ConversationalRetrievalChain, chunk: str, indicators: pd.DataFrame):
    try:
        prompt = f"""
        You are an AI assistant specialized in validating Red Cross reports. Generate 1 detailed validation question based on the following:

        Narrative Summary:
        {chunk}

        Indicator Information:
        {indicators.head().to_string()}  # Only send a sample of indicators to reduce payload

        Provide the question in this format:
        Country: [Country name]
        Created: {datetime.now().strftime('%Y-%m-%d')}
        Status: Open
        Section: [Relevant section from the data]
        Indicator: [Relevant indicator]
        Reported value: [Value from data or "Not Reported"]
        Question: [Your validation question]

        Focus on inconsistencies, gaps, and adherence to Red Cross reporting standards.
        """

        response = qa_chain({"question": prompt})
        return response['answer']
    except Exception as e:
        logger.error(f"Failed to generate question: {str(e)}")
        return f"Failed to generate question: {str(e)}"

def process_files(narrative_file, indicators_file, ns_data_file, financial_overview_file, bilateral_support_file, chunk_size, chunk_overlap):
    if all([narrative_file, indicators_file, ns_data_file, financial_overview_file, bilateral_support_file]):
        try:
            # Read and process narrative
            narrative_text = read_docx(narrative_file)
            narrative_chunks = chunk_text(narrative_text, chunk_size, chunk_overlap)
            
            # Create vector database from narrative chunks
            narrative_docs = [LangchainDocument(page_content=chunk, metadata={"source": "narrative"}) for chunk in narrative_chunks]
            vectordb = create_db(narrative_docs, "narrative_db")
            
            # Read other files
            indicators_data = read_excel(indicators_file)
            ns_data = read_excel(ns_data_file)
            financial_overview = read_excel(financial_overview_file)
            bilateral_support = read_excel(bilateral_support_file)
            
            # Initialize LLM and QA chain
            llm = initialize_llm()
            qa_chain = create_qa_chain(llm, vectordb)
            
            # Generate questions
            all_questions = []
            for i, chunk in enumerate(narrative_chunks):
                try:
                    question = generate_question(qa_chain, chunk, indicators_data)
                    all_questions.append(question)
                    time.sleep(random.uniform(1, 3))  # Add a random delay between API calls
                except Exception as e:
                    logger.error(f"Error generating question for chunk {i}: {e}")
                    all_questions.append(f"Failed to generate question for chunk {i}: {str(e)}")
            
            formatted_questions = format_questions("\n\n".join(all_questions))
            return formatted_questions
        
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
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

def create_demo():
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
        
        with gr.Accordion("Advanced options", open=False):
            chunk_size = gr.Slider(minimum=100, maximum=1000, value=250, step=50, label="Chunk size")
            chunk_overlap = gr.Slider(minimum=10, maximum=200, value=50, step=10, label="Chunk overlap")
        
        questions_output = gr.Textbox(label="Validation Questions", lines=10)
        
        generate_btn = gr.Button("Generate Validation Questions")
        generate_btn.click(fn=process_files, inputs=[narrative_file, indicators_file, ns_data_file, financial_overview_file, bilateral_support_file, chunk_size, chunk_overlap], outputs=questions_output)
        
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
    
    return demo

if __name__ == "__main__":
    # Configuration
    LOCAL_DEBUG = False  # Set to False when deploying to Hugging Face Spaces
    PORT = 7860  # Choose any available port

    # Create the Gradio demo
    demo = create_demo()

    if LOCAL_DEBUG:
        # Run locally
        demo.launch(server_port=PORT, share=False)
    else:
        # Deploy to Hugging Face Spaces
        demo.launch()