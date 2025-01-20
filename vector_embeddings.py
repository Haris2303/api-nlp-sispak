# RAG application built on gemini with LangChain Memory
# Memuat semua library yang diperlukan
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pandas as pd

# Memuat variabel lingkungan dari file .env
load_dotenv()
folder_path = r"D:\Dev\learn\python\nlp\rag\pdf"  # Ganti dengan path folder Anda

# Mengambil semua file PDF dalam folder
pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

all_data = []
for pdf_file in pdf_files:
    try:
        file_path = os.path.join(folder_path, pdf_file)
        loader = PyPDFLoader(file_path)
        data = loader.load() 
        all_data.extend(data)
        print(f"Berhasil memuat {len(data)} halaman dari {pdf_file}")
    except Exception as e:
        print(f"Kesalahan saat memuat PDF {pdf_file}: {e}")

if all_data:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(all_data)
    print(f"Jumlah total bagian dokumen: {len(docs)}")
else:
    print("Tidak ada data untuk diproses.")

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/bert-base-nli-max-tokens")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="data")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    print("Embeddings: ", len(embeddings))
    print("VectorStore: ", vectorstore)
    print("Retriever: ", vectorstore)
    print("Penyimpanan vektor berhasil dibuat dan disimpan.")
except Exception as e:
    print(f"Kesalahan saat menginisialisasi embeddings atau penyimpanan vektor: {e}")

# Menginisialisasi memory untuk percakapan
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None)

    # Menggunakan LangChain Memory pada prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "Anda adalah CatBot, sebuah Asisten diagnosa gangguan kesehatan mental mahasiswa yang siap membantu masalah gangguan mental mahasiswa dengan solusi yang tepat dan efektif."
                "Tugas Anda adalah mendiagnosa gejala yang disebutkan dengan cepat dan jelas, termasuk menyebutkan nama ilmiah gangguan kesehatan mental jika relevan sesuai dataset"
                "Anda menjelaskan penyebab masalah gangguan kesehatan mental dengan cara yang mudah dipahami."
                "Jawaban Anda harus ringkas, langsung ke inti dan tidak butuh data berupa gambar."
                "Anda berkomunikasi dengan ramah dan memberikan ketenangan agar Anda semangat mengatasi gangguan kesehatan mental mahasiswa."
                "Anda akan mengingat detail yang Anda sebutkan di percakapan kita."
                "Anda dikembangkan oleh Haris, mahasiswa Universitas Muhammadiyah Sorong Program Studi Teknik Informatika."
                "Sumber data yang digunakan berasal dari seorang pakar bernama Dr Yanto, Dokter Psikologi yang telah tersertifikasi di bidang Kesehatan Psikolog."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )
    
    # Mengintegrasikan memory ke dalam LLMChain
    conversation_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    # Mengintegrasikan conversation chain ke dalam RAG chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("RAG chain dengan memory berhasil diinisialisasi.")
except Exception as e:
    print(f"Kesalahan saat menginisialisasi RAG chain: {e}")

# Menguji query dan menampilkan similarity
try:
    query = "Saya sering mengalami tekanan di kampus seperti tugas yang sangat menumpuk, bagaimana saya bisa mengatasi hal tersebut?"
    
    # Mendapatkan dokumen yang relevan
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # Membuat respons dari model menggunakan memory
    response = conversation_chain.invoke({"input": query})  # Tidak ada ["answer"], langsung cetak respons
    print("Respons:", response)  # Menggunakan respons langsung
    
    # Menghitung embedding untuk query
    query_embedding = embeddings.embed_query(query)
    similarities = []
    
    for doc in retrieved_docs:
        doc_embedding = embeddings.embed_query(doc.page_content)
        similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        similarities.append((doc.page_content, similarity))
    
    # Mengurutkan hasil berdasarkan similarity
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    print("Hasil pencarian dengan tingkat kemiripan:")
    for content, score in similarities:
        print(f"Tingkat kemiripan: {score:.4f}")
    
except Exception as e:
    print(f"Kesalahan saat memproses query: {e}")

# Mengekspor objek penting untuk digunakan di modul lain
_all_ = ["retriever", "embeddings", "conversation_chain"]

if not (retriever and embeddings and conversation_chain):
    raise RuntimeError("Gagal menginisialisasi retriever, embeddings, atau conversation_chain.")

results = []
for i, (doc, similarity) in enumerate(zip(docs, similarities[0])):
    results.append({
        "Document": f"Document {i+1}",
        "Content": doc.page_content,
        "Similarity": similarity
    })
print("REsult: ", results)

# df = pd.DataFrame(results)
# df.to_csv("similarity_results.csv", index=False)

# embeddings_array = np.array([embeddings.embed_query(doc.page_content) for doc in docs])
# df_embeddings = pd.DataFrame(embeddings_array, columns=[f"Vector Dimension {i+1}" for i in range(embeddings_array.shape[1])])
# df_embeddings["Document"] = [f"Document {i+1}" for i in range(len(docs))]
# df_embeddings.to_excel("embeddings.xlsx", index=False)


# # Contoh embedding dari dokumen (gantikan dengan hasil sebenarnya)
# doc_embeddings = np.array([embeddings.embed_query(doc.page_content) for doc in docs])
# query_embedding = embeddings.embed_query(query)  # Embedding untuk query
# num_dimensions = doc_embeddings.shape[1]  # Jumlah dimensi embedding

# # Membuat DataFrame untuk menyimpan hasil
# data = []

# for i, (doc, doc_embedding) in enumerate(zip(docs, doc_embeddings)):
#     row = {
#         "Subjudul": f"Document {i+1}",
#         "Vector Dataset": doc.page_content,
#     }
#     # Tambahkan dimensi vektor dokumen
#     row.update({f"Dimensi {j+1}": doc_embedding[j] for j in range(num_dimensions)})
#     # Tambahkan kemiripan dengan query
#     row.update({f"Vector Embeddings {j+1}": query_embedding[j] for j in range(10)})  # Hanya 10 dimensi dari query
#     data.append(row)

# # Membuat DataFrame dari data
# df = pd.DataFrame(data)

# # Menyimpan DataFrame ke Excel
# output_file = "embedding_results.xlsx"
# df.to_excel(output_file, index=False, sheet_name="Embeddings")

# print(f"Data berhasil disimpan ke file Excel: {output_file}")



# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# # Contoh embedding dari dokumen (gantikan dengan hasil sebenarnya)
# doc_embeddings = np.array([embeddings.embed_query(doc.page_content) for doc in docs])
# query_embedding = embeddings.embed_query(query)  # Embedding untuk query
# num_dimensions = doc_embeddings.shape[1]  # Jumlah dimensi embedding

# # Menghitung cosine similarity
# cosine_similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

# print("Cosine Similiarities: DISINI DISINI!!! => ", cosine_similarities)
# print("INI ITU Total DOKUMENNYA LOHHH => ", len(cosine_similarities))

# # Membuat DataFrame untuk menyimpan hasil
# data = []

# for i, (doc, doc_embedding, similarity) in enumerate(zip(docs, doc_embeddings, cosine_similarities)):
#     row = {
#         "Subjudul": f"Document {i+1}",
#         "Vector Dataset": doc.page_content,
#         "Cosine Similarity": similarity, 
#     }
#     # dimensi vektor dokumen
#     row.update({f"Dimensi {j+1}": doc_embedding[j] for j in range(num_dimensions)})
#     # embedding query (hanya 10 dimensi pertama)
#     for j in range(10):
#         print(f"INI VECTOR EMBEDDINGS KE {j + 1} HEYY => ", query_embedding[j])
#     row.update({f"Vector Embeddings {j+1}": query_embedding[j] for j in range(10)})
#     data.append(row)

# # Membuat DataFrame dari data
# df = pd.DataFrame(data)

# # Menyimpan DataFrame ke Excel
# output_file = "embedding_cosine_similarity.xlsx"
# df.to_excel(output_file, index=False, sheet_name="Embeddings")

# print(f"Data berhasil disimpan ke file Excel: {output_file}")


# Menghitung embedding untuk setiap dokumen
# doc_embeddings = np.array([embeddings.embed_query(doc.page_content) for doc in docs])
# query_embedding = embeddings.embed_query(query)  # Embedding untuk query

# # Menghitung cosine similarity
# cosine_similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

# # Membuat DataFrame untuk menyimpan hasil
# data = []

# for i, (doc, doc_embedding, similarity) in enumerate(zip(docs, doc_embeddings, cosine_similarities)):
#     row = {
#         "Subjudul": f"Document {i+1}",
#         "Vector Dataset": doc_embedding.tolist(),  # Menyimpan vektor sebagai list
#         "Cosine Similarity": similarity, 
#     }
#     # Tambahkan dokumen asli ke data
#     row["Content"] = doc.page_content
#     data.append(row)

# # Membuat DataFrame dari data
# df = pd.DataFrame(data)

# # Menyimpan DataFrame ke Excel
# output_file = "embedding_cosine_similarity_with_vector_dataset.xlsx"
# df.to_excel(output_file, index=False, sheet_name="Embeddings")

# print(f"Data berhasil disimpan ke file Excel: {output_file}")

# Menghitung embedding untuk setiap dokumen
# doc_embeddings = np.array([embeddings.embed_query(doc.page_content) for doc in docs])
# query_embedding = embeddings.embed_query(query)  # Embedding untuk query
# num_dimensions = doc_embeddings.shape[1]

# # Menghitung cosine similarity
# cosine_similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

# # Membuat DataFrame untuk menyimpan hasil
# data = []

# for i, (doc, doc_embedding, similarity) in enumerate(zip(docs, doc_embeddings, cosine_similarities)):
#     row = {
#         "Subjudul": f"Document {i+1}",
#         "Content": doc.page_content,  # Konten dokumen
#         "Cosine Similarity": similarity,  # Similarity dengan query
#     }
#     # Menambahkan dimensi vektor dokumen
#     row.update({f"Vector Dataset {j+1}": doc_embedding[j] for j in range(num_dimensions)})
#     # Menambahkan 10 dimensi pertama dari query embedding
#     row.update({f"Vector Embeddings {j+1}": query_embedding[j] for j in range(10)})
#     data.append(row)

# # Membuat DataFrame dari data
# df = pd.DataFrame(data)

# # Menyimpan DataFrame ke Excel
# output_file = "embedding_cosine_similarity_detailed.xlsx"
# df.to_excel(output_file, index=False, sheet_name="Embeddings")

# print(f"Data berhasil disimpan ke file Excel: {output_file}")


# Mengurutkan hasil berdasarkan similarity
similarities_sorted = sorted(similarities, key=lambda x: x[1], reverse=True)

# Menyusun data untuk ditulis ke Excel
vector_embeddings = np.array([query_embedding])  # Misal embedding hasil model LLM
dataset_vectors = np.array([embeddings.embed_query(doc[0]) for doc in similarities_sorted])

# Menyusun data dalam bentuk dataframe
columns = ['Subjudul', 'Vector Dataset'] + [f'Vector Embeddings {i+1}' for i in range(len(similarities_sorted))]
rows = []

# Menyusun dimensi dan vektor ke dalam bentuk yang sesuai
for dim in range(len(vector_embeddings[0])):  # Misalkan dimensi 768
    row = ['Dimensi ' + str(dim+1), vector_embeddings[0][dim]]  # Vector Hasil Embedding
    for doc_vector in dataset_vectors:
        row.append(doc_vector[dim])  # Vector Dataset
    rows.append(row)

# Membuat DataFrame
df = pd.DataFrame(rows, columns=columns)

# Menulis DataFrame ke Excel
output_file = "embedding_manual_laporan.xlsx"
df.to_excel(output_file, index=False)

print(f"File Excel disimpan di {output_file}")