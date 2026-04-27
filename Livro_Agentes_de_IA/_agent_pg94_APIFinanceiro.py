from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage
import requests

from dotenv import load_dotenv

load_dotenv()

@tool
def get_stock_price(ticket: str) -> float:
    """Obter o preço da ação para o ticker da bolsa de valores da empresa"""
    api_url = f"https://api.example.com/stocks/{ticket}"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data['price']
    else:
        raise ValueError(f"Falha ao obter o preço da ação para {ticket}")

# Inicializar o LLM com o GPT-4o e vincular as ferramentas
llm = ChatOpenAI(model='gpt-4o', temperature=0)
llm_with_tools = llm.bind_tools([get_stock_price])

messages = [HumanMessage("Qual é o preço da ação da Apple?")]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    tool_msg = get_stock_price.invoke(tool_call)

    print(tool_msg.name)
    print(tool_call['args'])
    print(tool_msg.content)
    messages.append(tool_msg)
    print()

final_response = llm_with_tools.invoke(messages)
print(final_response.content)

