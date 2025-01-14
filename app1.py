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
from flask import Flask, request, jsonify

from dotenv import load_dotenv
load_dotenv()

# init flask
app = Flask(__name__)

file_path = ""

# init chromadb
client = chromadb.Client()

# make collection
collection = client.get_or_create_collection(name="chatbot_memory")

# chromadb.api.client.SharedSystemClient.clear_system_cache()

# st.title("Aplikasi AI Diagnonsi Gangguan Mental Mahasiswa")

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

# Langkah 1: Query data dari memori ChromaDB
# stored_docs = collection.query(
#     query_texts=[quetions],
#     n_results=5,  # Cari hingga 5 hasil relevan
#     where={"user_id": user_id},  # Cari berdasarkan user_id
# )

# print(stored_docs['documents'])

# context = "\n".join([doc['document'] for doc in docs])
# context = "\n".join([doc['document']])

# Langkah 2: Bangun konteks dengan informasi sebelumnya
# query = f"{quetions}\n\nInformasi sebelumnya yang diketahui:\n{context}"

def run_nlp(query):

    # Language gtts is indonesian
    # language = "id"
    # file_path = ""

    # player = pyglet.media.Player()
    # response = {"answer": ""}

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
            #print(response["answer"])
            # sentence = response['answer']

            # embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001").embed_documents([sentence])
            # padded_embedding = pad_or_truncate_embedding(embedding_model[0])

            # Langkah 3: Simpan informasi baru jika relevan
            # if "ingat" in quetions.lower() or "simpan" in quetions.lower():
            # collection.add(
            #     ids=[f"{user_id}_{time.time()}"],  # Buat ID unik
            #     documents=[sentence],
            #     metadatas=[{"user_id": user_id}],
            #     embeddings=[padded_embedding]
            # )

            # st.write(response["answer"])
            # file = gTTS(text=sentence, lang=language, slow=False)
            # file.save("audio/audio.mp3") 

            # file_path = "audio/audio.mp3"
            # src = pyglet.media.load(file_path)
            # player.queue(src)

            return response
        except Exception as err:
            print(f"Error: {str(err)}")

    # return response

# st.write(response['answer'])
# player.play()

# Endpoint untuk GET
@app.route('/api', methods=['GET'])
def get_data():
    req = request.args.get('quetion')
    # user_id = request.args.get('user_id')
    response = run_nlp(req)
    print(response)
    return jsonify({"message": response})

# try remove file audio
try: 
    os.remove(file_path)
    print(f"File '{file_path}' deleted successfully.")

except FileNotFoundError: print(f"File '{file_path}' not found.")

if __name__ == '__main__':
    app.run(debug=True)