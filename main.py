import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
 
load_dotenv()
 
st.header("HTML CHATBOT")

html_file = st.file_uploader("Upload HTML", type='html')

if html_file is not None:
    data = html_file.read().decode('utf-8')
 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splitted_text = text_splitter.split_text(text=data)
 
    vector_embedding = OpenAIEmbeddings()
    vector_database = FAISS.from_texts(splitted_text, embedding=vector_embedding)
 
    question = st.text_input("Ask a question:")
 
    if question:

        llm = OpenAI()
        retriever_qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type='stuff',
            retriever=vector_database.as_retriever(),
            )
        answer = retriever_qa.run(question)
        st.write(answer)
 
