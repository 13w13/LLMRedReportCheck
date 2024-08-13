import gradio as gr
import pandas as pd
from docx import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceEndpoint
from typing import List, Dict, Tuple
import re
import os
import chromadb
from pathlib import Path
from unidecode import unidecode
import logging
import spaces

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the Llama 3.1 model
MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Get API token from environment variable
HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

def extract_text_from_docx(file) -> Tuple[str, Dict[int, int]]:
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

def load_excel_data(file: gr.File) -> pd.DataFrame:
    return pd.read_excel(file.name, engine='openpyxl')

@spaces.GPU
def create_vector_db(text: str, collection_name: str) -> Chroma:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding,
        client=new_client,
        collection_name=collection_name
    )
    return vectordb

@spaces.GPU
def generate_validation_question(llm: HuggingFaceEndpoint, indicator: str, document_section: str, reported_value: str) -> str:
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant specialized in validating Red Cross reports. Analyze the following information and generate a detailed validation question.

Indicator: {indicator}
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
    
    response = llm(prompt)
    return response.strip()

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

@spaces.GPU(duration=300)  # Set a higher duration if needed
def process_files(narrative_file: gr.File, ns_data_file: gr.File, indicators_file: gr.File, finance_file: gr.File, support_file: gr.File) -> str:
    logging.info("Starting process_files function")
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_NAME,
        temperature=0.7,
        max_new_tokens=150,
        top_k=50,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

    narrative_text, page_mapping = extract_text_from_docx(narrative_file)
    
    excel_data = {
        "NS Data": load_excel_data(ns_data_file) if ns_data_file else None,
        "Indicators": load_excel_data(indicators_file) if indicators_file else None,
        "Finance": load_excel_data(finance_file) if finance_file else None,
        "Support": load_excel_data(support_file) if support_file else None
    }
    
    collection_name = create_collection_name(narrative_file.name)
    vectordb = create_vector_db(narrative_text, collection_name)
    
    results = []

    for section, df in excel_data.items():
        if df is None:
            logging.warning(f"Section {section} not provided")
            continue
        
        if section == "Indicators":
            for _, row in df.iterrows():
                indicator = row.get('Indicator', 'Unknown Indicator')
                reported_value = row.get('2024 Midyear', 'N/A')
                
                query = indicator
                similar_chunks = vectordb.similarity_search(query, k=2)
                
                if similar_chunks:
                    for chunk in similar_chunks:
                        source_text = chunk.page_content
                        start_index = narrative_text.index(source_text)
                        page_number = get_page_number(start_index, page_mapping)
                        
                        validation_question = generate_validation_question(llm, indicator, source_text, reported_value)
                        
                        results.append({
                            'Section': section,
                            'Indicator': indicator,
                            'Reported Value': reported_value,
                            'Source from Doc': source_text,
                            'Page Number': page_number,
                            'Validation Question': validation_question
                        })
                else:
                    logging.warning(f"No similar chunks found for indicator: {indicator}")
                    validation_question = generate_validation_question(llm, indicator, "Indicator not found in narrative", reported_value)
                    results.append({
                        'Section': section,
                        'Indicator': indicator,
                        'Reported Value': reported_value,
                        'Source from Doc': "Indicator not found in narrative",
                        'Page Number': "N/A",
                        'Validation Question': validation_question
                    })
        elif section == "Support":
            for _, row in df.iterrows():
                national_society = row.get('National Society name', 'Unknown NS')
                sp1_status = row.get('SP1 - Climate and enviroment', 'N/A')
                sp2_status = row.get('SP2 - Disasters and crises', 'N/A')
                sp3_status = row.get('SP3 - Health and wellbeing', 'N/A')
                
                validation_question = generate_validation_question(llm, f"Support status for {national_society}", 
                                                                   f"SP1: {sp1_status}, SP2: {sp2_status}, SP3: {sp3_status}", 
                                                                   "N/A")
                
                results.append({
                    'Section': section,
                    'Indicator': f"Support status for {national_society}",
                    'Reported Value': f"SP1: {sp1_status}, SP2: {sp2_status}, SP3: {sp3_status}",
                    'Source from Doc': "Support data",
                    'Page Number': "N/A",
                    'Validation Question': validation_question
                })
        else:
            # Handle NS Data and Finance sections if needed
            pass
    
    results_str = "Validation Results:\n\n"
    for result in results:
        results_str += f"Section: {result['Section']}\n"
        results_str += f"Indicator: {result['Indicator']}\n"
        results_str += f"Reported Value: {result['Reported Value']}\n"
        results_str += f"Source from Doc: {result['Source from Doc'][:100]}...\n"
        results_str += f"Page Number: {result['Page Number']}\n"
        results_str += f"Validation Question: {result['Validation Question']}\n\n"
    
    logging.info(f"Generated results: {results_str[:500]}...")  # Log first 500 characters of results
    return results_str

demo = gr.Interface(
    fn=process_files,
    inputs=[
        gr.File(label="Narrative Document (Word)"),
        gr.File(label="NS Data Excel File"),
        gr.File(label="Indicators Excel File"),
        gr.File(label="Finance Excel File"),
        gr.File(label="Support Excel File"),
    ],
    outputs=gr.Textbox(label="Validation Results", lines=20),
    title="Red Cross Report Validator",
    description="Upload files to generate validation questions."
)

if __name__ == "__main__":
    demo.launch()