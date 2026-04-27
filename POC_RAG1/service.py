from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
import pandas as pd
import numpy as np
from langchain_core.documents import Document

from dotenv import load_dotenv
from lxml.html import document_fromstring

load_dotenv()

CAMINHO_DB = "C:\\Users\\Nitro\\Desktop\\Xip\\POC_RAG1\\db.xlsx"

def cria_db_vetor():
    documentos = ler_db(CAMINHO_DB)
    chunks = criar_chunks(documentos)
    vetorizar_chunks(chunks)
    return

def ler_db(caminho):
    db = pd.read_excel(CAMINHO_DB)
    corpo = [db.iloc[x:x+1] for x in range(len(db))]
    corpo = [str(x) for x in corpo]
    cabecario = corpo[0].split('18:00:00')[0] + "18:00:00"
    corpo = list(map(lambda x: x.split('18:00:00')[-1], corpo))
    corpo.insert(0,cabecario)
    # corpo = "\n___\n".join(corpo)
    print(type(corpo))
    return corpo

# ler_db(CAMINHO_DB)

def criar_chunks(documentos):
    docs = [Document(page_content=texto) for texto in documentos]
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        length_function=len,
        add_start_index=True
    )
    chunks = separador_documentos.split_documents(docs)
    print(len(chunks))
    return chunks

def vetorizar_chunks(chunks):
    Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory='db')
    print("Banco de dados criado")

#cria_db_vetor()
if __name__ == '__main__':
    #    ler_db(CAMINHO_DB)
    pass






