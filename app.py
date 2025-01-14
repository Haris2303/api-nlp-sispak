import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
import chromadb
import pyglet
from gtts import gTTS
import os 
from flask import Flask, request, jsonify, Response

from dotenv import load_dotenv
load_dotenv()

# init flask
app = Flask(__name__)

file_path = ""

loader = PyPDFLoader("document.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

system_prompt = (
    "Anda adalah asisten untuk tugas-tugas menjawab pertanyaan. "
    "Dan dapat mengingat informasi pribadi pengguna. "
    "Jika pengguna meminta Anda untuk mengingat informasi, simpanlah ke dalam memori. "
    "Gunakan potongan konteks yang diambil berikut ini untuk menjawab "
    "pertanyaannya. Jika Anda tidak tahu jawabannya, katakan bahwa Anda "
    "tidak tahu. Gunakan maksimal tiga kalimat dan jaga agar "
    "jawablah dengan ringkas."
    "\n\n"
    "{context}"
    "Riwayat chat sebelumnya dari user: {chat_history}"
)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{quetion}")
    ]
)

# Init conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="quetion"
)

def run_nlp(query):
    if query:
        try:
            retrieved_docs = retriever.get_relevant_documents(query)
            context = "\n" . join(doc.page_content for doc in retrieved_docs)

            conversation = LLMChain(
                llm=llm,
                prompt=prompt,
                verbose=True,
                memory=memory
            )

            response = conversation.predict(
                quetion=query,
                context=context
            )

            chat_history = memory.load_memory_variables({})['chat_history']

            return response
        except Exception as err:
            print(f"Error: {str(err)}")

# Endpoint untuk GET
@app.route('/api', methods=['GET'])
def get_data():
    req = request.args.get('quetion')
    response = run_nlp(req)
    print(response)
    res = Response(
        response=jsonify({"message": response}).get_data(as_text=True),
        status=200,
        mimetype="application/json"
    )
    return res

# try remove file audio
try: 
    os.remove(file_path)
    print(f"File '{file_path}' deleted successfully.")

except FileNotFoundError: print(f"File '{file_path}' not found.")

if __name__ == '__main__':
    app.run(debug=True)