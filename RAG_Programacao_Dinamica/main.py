from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import  ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

CAMINHO_DB = 'db'

promp_template ="""
Responda a pergunta do usuário:
 {pergunta}
 
 com base nessas informações:
 {base_conhecimento}
 
 """

def perguntar():
    pergunta = input("Escreva sua pergunta: ")

    # carregar o banco de dados
    funcao_embedding = OpenAIEmbeddings()
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)

    # comparar a pergunta do usuário (embedding) com o meu banco de dados
    resultados = db.similarity_search_with_relevance_scores(pergunta, k=3)
    if len(resultados) == 0 or resultados[0][1] < 0.7:
        print("Não conseguiu encontrar alguma informação relevante")
        return

    textos_resultados =[]
    for resultado in resultados:
        texto = resultado[0].page_content
        textos_resultados.append(texto)

    base_conhecimento = "\n\n----\n\n".join(textos_resultados)
    prompt = ChatPromptTemplate.from_template(promp_template)
    prompt = prompt.invoke({"pergunta" : pergunta, "base_conhecimento" : base_conhecimento})

    modelo = ChatOpenAI()
    texto_resposta = modelo.invoke(prompt).content
    print("Resposta da IA: ", texto_resposta)

perguntar()

