import gradio as gr
import pandas as pd
from docx import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceEndpoint
import torch
from typing import List, Dict, Tuple
import re
import os
import chromadb
from pathlib import Path
from unidecode import unidecode
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the Llama 3.1 model
MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Get API token from environment variable
HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

def load_indicators(file) -> Dict[str, Dict]:
    try:
        df = pd.read_excel(file.name, engine='openpyxl')
        return {str(row['KPI ID']): row.to_dict() for _, row in df.iterrows()}
    except Exception as e:
        logging.error(f"Error loading indicators: {str(e)}")
        raise gr.Error(f"Error loading indicators: {str(e)}")

def extract_text_from_docx(file) -> Tuple[str, Dict[int, int]]:
    try:
        doc = Document(file.name)
        full_text = ""
        page_mapping = {}
        current_page = 1
        words_on_page = 0
        
        for para in doc.paragraphs:
            full_text += para.text + "\n"
            words = len(para.text.split())
            words_on_page += words
            
            if words_on_page > 300:  # Approximate words per page
                page_mapping[len(full_text)] = current_page
                current_page += 1
                words_on_page = 0
        
        return full_text, page_mapping
    except Exception as e:
        logging.error(f"Error extracting text from document: {str(e)}")
        raise gr.Error(f"Error extracting text from document: {str(e)}")

def load_excel_data(files: List[gr.File]) -> Dict[str, pd.DataFrame]:
    try:
        return {file.name.split('/')[-1].split('.')[0]: pd.read_excel(file.name, engine='openpyxl') for file in files}
    except Exception as e:
        logging.error(f"Error loading Excel data: {str(e)}")
        raise gr.Error(f"Error loading Excel data: {str(e)}")

def create_vector_db(text: str, collection_name: str) -> Chroma:
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        
        embedding = HuggingFaceEmbeddings()
        new_client = chromadb.EphemeralClient()
        vectordb = Chroma.from_texts(
            texts=chunks,
            embedding=embedding,
            client=new_client,
            collection_name=collection_name
        )
        return vectordb
    except Exception as e:
        logging.error(f"Error creating vector database: {str(e)}")
        raise gr.Error(f"Error creating vector database: {str(e)}")

def generate_validation_question(llm: HuggingFaceEndpoint, indicator: Dict, document_section: str, reported_value: str) -> str:
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant specialized in validating Red Cross reports. Analyze the following information and generate a detailed validation question.

Indicator: {indicator['Indicator Name']}
Definition: {indicator['Definition']}
Reported Value: {reported_value}
Relevant Document Section:
{document_section[:1000]}  # Truncate to avoid exceeding token limit

Focus on the following aspects:
1. Inconsistencies between the reported value and the narrative
2. Gaps in information or missing details
3. Adherence to Red Cross reporting standards
4. Potential misreporting or misinterpretation
5. Suggestions for additional data or clarification needed

If there are no apparent issues, suggest a question to verify the accuracy and completeness of the reported information.<|eot_id|><|start_header_id|>user<|end_header_id|>
Generate a validation question based on the provided information.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Question: """
    
    try:
        response = llm(prompt)
        return response.strip()
    except Exception as e:
        logging.error(f"Error generating validation question: {str(e)}")
        return f"Error generating validation question: {str(e)}"

def get_page_number(index: int, page_mapping: Dict[int, int]) -> int:
    for char_index, page in sorted(page_mapping.items(), reverse=True):
        if index >= char_index:
            return page
    return 1

def create_collection_name(filepath):
    collection_name = Path(filepath).stem
    collection_name = collection_name.replace(" ","-") 
    collection_name = unidecode(collection_name)
    collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)
    collection_name = collection_name[:50]
    if len(collection_name) < 3:
        collection_name = collection_name + 'xyz'
    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'
    return collection_name

def process_files(indicators_file: gr.File, narrative_file: gr.File, excel_files: List[gr.File], selected_sections: List[str], progress=gr.Progress()) -> pd.DataFrame:
    progress(0, desc="Initializing...")
    logging.info("Starting file processing")
    
    # Initialize LLM
    try:
        llm = HuggingFaceEndpoint(
            repo_id=MODEL_NAME,
            temperature=0.7,
            max_new_tokens=150,
            top_k=50,
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        )
    except Exception as e:
        logging.error(f"Error initializing LLM: {str(e)}")
        raise gr.Error(f"Error initializing LLM: {str(e)}")

    progress(0.1, desc="Loading indicators...")
    indicators = load_indicators(indicators_file)
    progress(0.2, desc="Extracting narrative...")
    narrative_text, page_mapping = extract_text_from_docx(narrative_file)
    progress(0.3, desc="Loading Excel data...")
    excel_data = load_excel_data(excel_files)
    
    progress(0.4, desc="Creating vector database...")
    collection_name = create_collection_name(narrative_file.name)
    vectordb = create_vector_db(narrative_text, collection_name)
    
    results = []
    total_steps = sum(len(excel_data[section]) for section in selected_sections if section in excel_data)
    current_step = 0

    for section in selected_sections:
        if section not in excel_data:
            logging.warning(f"Section {section} not found in Excel data")
            continue
        
        df = excel_data[section]
        
        for _, row in df.iterrows():
            current_step += 1
            progress(0.4 + (0.6 * current_step / total_steps), desc=f"Processing {section} - {current_step}/{total_steps}")
            
            indicator_id = str(row.get('KPI ID', ''))
            if indicator_id in indicators:
                indicator = indicators[indicator_id]
                reported_value = row.get('Value', 'N/A')
                
                query = f"{indicator['Indicator Name']} {indicator['Definition']}"
                similar_chunks = vectordb.similarity_search(query, k=2)
                
                if similar_chunks:
                    for chunk in similar_chunks:
                        source_text = chunk.page_content
                        start_index = narrative_text.index(source_text)
                        page_number = get_page_number(start_index, page_mapping)
                        
                        validation_question = generate_validation_question(llm, indicator, source_text, reported_value)
                        
                        results.append({
                            'Section': section,
                            'Indicator': indicator['Indicator Name'],
                            'Reported Value': reported_value,
                            'Source from Doc': source_text,
                            'Page Number': page_number,
                            'Validation Question': validation_question
                        })
                else:
                    logging.warning(f"No similar chunks found for indicator {indicator['Indicator Name']}")
                    validation_question = generate_validation_question(llm, indicator, "Indicator not found in narrative", reported_value)
                    results.append({
                        'Section': section,
                        'Indicator': indicator['Indicator Name'],
                        'Reported Value': reported_value,
                        'Source from Doc': "Indicator not found in narrative",
                        'Page Number': "N/A",
                        'Validation Question': validation_question
                    })
            else:
                logging.warning(f"Indicator ID {indicator_id} not found in indicators list")
        
        # Check for indicators in the narrative that are not reported
        for indicator_id, indicator in indicators.items():
            if indicator_id not in df['KPI ID'].astype(str).values:
                query = f"{indicator['Indicator Name']} {indicator['Definition']}"
                similar_chunks = vectordb.similarity_search(query, k=1)
                
                if similar_chunks:
                    chunk = similar_chunks[0]
                    source_text = chunk.page_content
                    start_index = narrative_text.index(source_text)
                    page_number = get_page_number(start_index, page_mapping)
                    
                    validation_question = generate_validation_question(llm, indicator, source_text, "Not Reported")
                    
                    results.append({
                        'Section': section,
                        'Indicator': indicator['Indicator Name'],
                        'Reported Value': 'Not Reported',
                        'Source from Doc': source_text,
                        'Page Number': page_number,
                        'Validation Question': validation_question
                    })
                else:
                    logging.warning(f"No similar chunks found for unreported indicator {indicator['Indicator Name']}")
    
    progress(1.0, desc="Completed!")
    logging.info(f"Processing completed. Generated {len(results)} validation questions.")
    return pd.DataFrame(results)

def demo():
    with gr.Blocks(theme="base") as demo:
        gr.Markdown("# Red Cross Report Validator")
        gr.Markdown("Upload files and select sections to generate validation questions.")
        
        with gr.Row():
            indicators_file = gr.File(label="Indicators List (Excel)")
            narrative_file = gr.File(label="Narrative Document (Word)")
            excel_files = gr.File(label="Excel Files", file_count="multiple")
        
        sections = gr.CheckboxGroup(choices=["NS Data", "Indicators", "Finance", "Support"], label="Select Sections")
        
        submit_btn = gr.Button("Generate Validation Questions")
        
        output = gr.DataFrame(label="Validation Results")
        
        submit_btn.click(process_files, inputs=[indicators_file, narrative_file, excel_files, sections], outputs=output)
    
    return demo

if __name__ == "__main__":
    demo()