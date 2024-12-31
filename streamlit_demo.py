import streamlit as st

import os
import pandas as pd
from tqdm.auto import tqdm
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pytz import timezone

# OpenAI + Langchain
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder)
from langchain.chains import RetrievalQA, LLMChain,ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

import PyPDF2
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import CharacterTextSplitter
import tempfile
import warnings
warnings.filterwarnings('ignore')

from utils.config import DATA_PATH, DB_DIRECTORY, MODEL_NAME

with open(os.path.join(DATA_PATH, 'photosyn_qa_prompt_history.pickle'), 'rb') as f:
    prompt_history = pickle.load(f)

model_name = MODEL_NAME
db_directory = DB_DIRECTORY

guideline_prompt = prompt_history[-1]


sys_prompt: PromptTemplate = PromptTemplate(
    template= """CHAT HISTORY: {chat_history}
    """+f"""{guideline_prompt}""",
    input_variables=["chat_history"],
)

system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

student_prompt: PromptTemplate = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    Question: {question}
    ---
    Retrieved Context : {context}
    Given the context information and not prior knowledge, answer the question. 
    ---
    Answer:"""
)

student_message_prompt = HumanMessagePromptTemplate(prompt=student_prompt)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, student_message_prompt])

memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key = 'answer')


llm = ChatOpenAI(model_name=model_name, temperature=0)
embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=db_directory, embedding_function=embeddings)
retriever = vectordb.as_retriever()


qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    return_source_documents=True,
    verbose=False,
    condense_question_llm = llm,
    chain_type="stuff",
    get_chat_history=lambda h : h,
    combine_docs_chain_kwargs={"prompt": chat_prompt},
    memory = memory
)

import streamlit as st

st.title('🌿 Photosynthesis Research Assistant')

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.sources = []

def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        file_path = tmp_file.name

    loader = PyPDFLoader(file_path)
    document = loader.load()

    os.remove(file_path)
    return document


# PDF file upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    
    inserted_document = process_pdf(uploaded_file)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_document = text_splitter.split_documents(inserted_document)

    # Add unique IDs to each chunk
    for i, chunk in enumerate(chunked_document):
        chunk.metadata['id'] = f"chunk_{i}"
    
    vectordb.add_documents(chunked_document)


    #if save: vertordb.persist()
    retriever = vectordb.as_retriever()
    qa.retriever = retriever

    st.success(f"Uploaded and processed {uploaded_file.name}")
    

user_input = st.text_input("Ask a question about photosynthesis:", key="user_query")


if st.button('Ask'):
    # chatbot response
    response = qa.invoke({'question': user_input})

    # fix the source information
    src_docs = []
    for i in range(len(response['source_documents'])):
        if response['source_documents'][i].metadata['source'].startswith('/home/jwoosang1/'):
            src_ = response['source_documents'][i].metadata['source'].replace('/home/jwoosang1/Project/llm/photosyn_llm/data/photosyn_llm_papers_150/','').replace('.pdf','').split(' ')[1:]
            src = '_'.join(src_)
            src_docs.append(src)
        else:
            src_docs.append(response['source_documents'][i].metadata['source'].replace('.pdf',''))
                            
    # save the response and source info.
    answer = response['answer']
    source_content = []
    
    for i in range(len(src_docs)):
        source_content.append(f'''{response['source_documents'][i].page_content.strip()}''')
    
    # add the recent response to chat history
    conversation_idx = len(st.session_state.chat_history)  # 새로운 대화의 인덱스  
    st.session_state.chat_history.insert(0, [("User", user_input, conversation_idx), ("Chatbot", answer, conversation_idx)])
    
    st.session_state.sources.insert(0, (src_docs, source_content, conversation_idx))

# print on interface
for idx, conversation in enumerate(st.session_state.chat_history):
    for role, text, conv_idx in conversation:
        title = "💬User:" if role == "User" else "🤖Chatbot:"
        height = 1 if role == "User" else max(300, len(text) // 30)
        key = f"history_{conv_idx}_{role.lower()}"  # 怨좎쑀??key ?앹꽦
        st.text_area(title, value=text.strip(), height=height, key=key)
        if role == "Chatbot":
            if st.button("🔗Source", key=f"source_{conv_idx}"):
                src_docs, contents, _ = next((src, cont, i) for src, cont, i in st.session_state.sources if i == conv_idx)
                for i, (src, content) in enumerate(zip(src_docs, contents)):
                    st.sidebar.markdown(f"### 📑 RAG Source {i+1}\n`{src}`")
                    st.sidebar.markdown("### 📑 Extracted Contents")
                    st.sidebar.markdown(f"```{content}```")