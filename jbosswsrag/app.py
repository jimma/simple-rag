from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from flask import Flask,render_template,request,jsonify

from typing import Dict, List, Any
from llama_cpp import Llama
import os
import logging
from transformers import AutoModel, AutoTokenizer

template_dir = os.path.abspath('./app/html/')
static_dir = os.path.abspath('./app/static/')
app = Flask(__name__, template_folder=template_dir, static_url_path='', static_folder=static_dir)
app.logger.setLevel(logging.INFO)

system_prompt = """
  You are an intelligent assistant. You will be provided with some context, as well as the question.
  Your job is to understand the question, and answer based on the things you already know and the context I provided.
"""


def load_pdf_to_vectorstore(pdf_folder_path) : 
    loader = PyPDFDirectoryLoader(pdf_folder_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 40)
    all_splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=HuggingFaceEmbeddings())
    return vectorstore

def construct_prompt(system_prompt, retrieved_docs, question):
    prompt = f"""{system_prompt}

    Context:
    {retrieved_docs}

    Question:
    {question}
    """
    return prompt

vector_store=load_pdf_to_vectorstore("./app/pdf")
llm=Llama(model_path="./app/model/mistral-7b-instruct-v0.2.Q3_K_L.gguf")

@app.route('/')
def index():
    return render_template('index.htm')


@app.route('/ask', methods=['GET', 'POST'])
def promptRequst():
    app.logger.info('Incoming http request')
    # Process the request
    question = request.get_json()["question"]
    if not question:
        return jsonify({"error":"missing required parameter 'question'"})
    else :
        app.logger.info("question is " + question)

    retrieved_docs = vector_store.similarity_search(question, k=1)[0].page_content


    prompt = construct_prompt(system_prompt=system_prompt, retrieved_docs=retrieved_docs, question=question)

    app.logger.info(f'constructed prompt: {prompt}')

    formatted_prompt = f"Q: {prompt} A: "
    
    
    response = llm(formatted_prompt, max_tokens=2048, stop=["Q:", "\n"], echo=False, stream=False)
    
    if not response['choices'][0]['text']:
       response['choices'][0]['text']="Sorry, I don't have the answer now. Please try asking the question in a different way and provide me with more details." 
                   
    app.logger.info(f'**** response **** : {response}')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)