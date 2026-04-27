from langchain_core.messages import HumanMessage
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

# Definir ferramentas usando definições concisas ds função
@tool
def multiply(x: float, y: float) -> float:
    """Multiplica 'x' por 'y'"""
    return x * y

@tool
def exponencialmente(x: float, y:float) -> float:
    """Elevar 'x' à potência 'y'"""
    return x ** y

@tool
def add(x: float, y: float) -> float:
    """Somar 'x' e 'y'"""
    return x + y

tools = [multiply, exponencialmente, add]

# Inicializar o LLM com GPT-4o e vincular as ferramentas
llm = ChatOpenAI(model='gpt-4o', temperature=0)
llm_with_tools = llm.bind_tools(tools)

query = "Quanto é 393 * 12.25? Além disso, quanto é 11 + 49?"
messages = [HumanMessage(query)]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply" : multiply, "exponencialmente" :  exponencialmente}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)

print(f"{tool_msg.name} {tool_call['args']} {tool_msg.content}")
messages.append(tool_msg)
final_response = llm_with_tools.invoke(messages)
print(final_response.content)

