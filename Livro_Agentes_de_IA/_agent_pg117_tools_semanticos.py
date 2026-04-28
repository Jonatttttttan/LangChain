import os
import requests
import logging
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

# Inicializar as incorporações OpenAI
embeddings = OpenAIEmbeddings(api_key = API_KEY)

# Descrição da ferramenta
tool_descriptions = {
    "query_wolfram_alpha" : '''Use o Wolfram para computar expressões matemáticas ou recuperar informações''',
    "trigger_zapier_webhook" : '''Acione um webhook do Zapier para executar fluxos de trabalho automatizados predefinidos''',
    "send_slack_message" : '''Envie mensagens para canais específicos do Slack para se comunicar com os membros da equipe'''
}

# TOOLS
@tool
def query_wolfram_alpha(entrada: str) -> float:
    """Use o Wolfram para computar expressões matemáticas ou recuperar informações"""
    return f"Chamando query_wolfram_alpha com a entrada {entrada}"

@tool
def trigger_zapier_webhook(entrada: str) -> str:
    """Acione um webhook do Zapier para executar fluxos de trabalho automatizados predefinidos"""
    return f"Chamando trigger_zapier_webhook com a entrada {entrada}"

@tool
def send_slack_message(entrada: str) -> str:
    """Envie mensagens para canais específicos do Slack para se comunicar com os membros da equipe"""
    return f"Chamando send_slack_message com a entrada {entrada}"


# Criar incorporações para cada descrição de ferramenta
tool_embeddings = []
tool_names = []

for tool_name, description in tool_descriptions.items():
    embedding = embeddings.embed_query(description)
    tool_embeddings.append(embedding)
    tool_names.append(tool_name)

# Inicializar o repositório vetorial FAISS
dimension = len(tool_embeddings[0])
index = faiss.IndexFlatL2(dimension)

# Normalizar incorporações para similaridade por cosseno
faiss.normalize_L2(np.array(tool_embeddings).astype('float32'))

# Converter lista para formato compatível com FAISS
tool_embeddings_np = np.array(tool_embeddings).astype('float32')
index.add(tool_embeddings_np)

# Mapear índice para funções da ferramenta
index_to_tool = {
    0: query_wolfram_alpha,
    1: trigger_zapier_webhook,
    2: send_slack_message
}

llm = ChatOpenAI(model='gpt-4o', temperature=0)
def select_tool(query: str, top_k: int = 1) -> list:
    """
    Seleciona a(s) ferramenta(s) mais relevante(s) com base na
    consulta do usuário usando recuperação baseada em vetores.

    Args:
        query (str): A consulta de entrada do usuário.
        top_k (int): Número de ferramentas principais a serem recuperadas.

    Returns:
        list: Lista das funções de ferramentas selecionadas.
    """

    query_embedding = embeddings.embed_query(query).astype('float32')
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    D, I = index.search(query_embedding.reshape(1, -1), top_k)
    selected_tools = [index_to_tool[idx] for idx in I[0] if idx in index_to_tool]
    return selected_tools

def determine_parameters(query: str, tool_name: str) -> dict:
    """
    Utiliza o LLM para analisar a consulta e determinar os
    parâmetros da ferramenta a ser invocada.

    Args:
        query (str): A consulta de entrada do usuário.
        tool_name (str): O nome da ferramenta selecionada.

    Returns:
        dict: Parâmetros para a ferramenta.
    """
    messages = [
        HumanMessage(content=f'''Com base na consulta do usuário: '{query}',
           quais parâmetros devem ser usados para a ferramenta '{tool_name}'?''')
    ]
    # Chamar o LLM para extrair parâmetros
    response = llm(messages)

    # Exemplo de lógica para analisar a resposta do LLM
    parameters = []
    if tool_name == 'query_wolfram_alpha':
        parameters['expression'] = response['expression']
        # Extrair expressão matemática
    elif tool_name == 'trigger_zapier_webhook':
        parameters['zap_id'] = response.get('zap_id', '123456')
        parameters['payload'] = response.get('payload', {"data" : query})
    elif tool_name == 'send_slack_message':
        parameters['channel'] = response.get('channel', '#general')
        parameters['message'] = response.get('message' , query)
    return parameters

# Exemplo de consulta do usuário
user_query = "Resolva esta equação 2x + 3 = 7"

# Selecionar a ferramenta da parte superior
selected_tools = select_tool(user_query, top_k=1)
tool_name = selected_tools[0] if selected_tools else None

if tool_name:
    # Usar LLM para determinar os parâmetros com base
    # na consulta e na ferramenta selecionada
    args = determine_parameters(user_query, tool_name)

    # Invocar a ferramenta selecionada
    try:
        # Assumindo que cada ferramenta tem um método 'invoke' para executá-la
        tool_result = globals()[tool_name].invoke(args)
        print(f"Resultado da ferramenta '{tool_name}' : {tool_result}")
    except ValueError as e:
        print(f"Erro ao invocar a ferramenta '{tool_name}' : {e}")
else:
    print("Nenhuma ferramenta foi selecionada")





