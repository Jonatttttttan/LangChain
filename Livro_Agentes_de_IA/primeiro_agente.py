from typing import TypedDict, Any

from langchain.tools import tool
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage # antigo langchain.schema...import
from langchain_core.messages.tool import ToolMessage
from langgraph.graph import StateGraph, START, END

import dotenv


dotenv.load_dotenv()

# -- 1) Definir nossa ferramenta de negócios única
@tool
def cancel_order(order_id: str) -> str:
    """Cancelar um pedido que ainda não foi enviado"""
    #(Aqui você chamaria sua verdadeira API de backend)
    return f"O pedido {order_id} foi cancelado"

# Schema do estado do grafo
class AgentState(TypedDict):
    order: dict[str, Any]
    messages: list

# -- 2) O agente "brain": invocar LLM, executar ferramenta e invocar LLM novamente
def call_model(state: AgentState):
    msgs = state["messages"]
    order = state.get("order", {"order_id" : "NÃO_IDENTIFICADO"})

    # Prompt do sistema diz ao modelo exatamente o que fazer
    prompt = (
        f'''Você é um agente de suporte de e-commerce.
        ORDER ID: {order["order_id"]}
        Se o cliente pedir para cancelar, chame cancel_order(order_id)
        e então envie uma confirmação simples.
        Caso contrário, apenas responda normalmente.'''
        ).strip()
    model = ChatOpenAI(model="gpt-5", temperature=0).bind_tools([call_model])


    full = [SystemMessage(content=prompt)] + msgs

    # Primeira passagem pelo LLM: decide chamar ou não nossa ferramente
    first = model.invoke(full)
    out = [first]

    if getattr(first, "tool_calls", None):
        # Executa a ferramenta cancel_order
        tc = first.tool_calls[0]

        # Executa a ferramenta com os argumentos que o modelo gerou
        result = cancel_order.invoke(tc["args"])

        # Devolve o resultado da ferramenta ao modelo
        out.append(
            ToolMessage(
                content=result,
                tool_call_id=tc["id"],
            )
        )



        # 2° passagem pelo LLM: gera o texto final de confirmação
        second = model.invoke(full+out)
        out.append(second)
    return {"messages": out}

# -- 3) Conecta tudo em um StateGraph
def construct_graph():
    g = StateGraph(AgentState)
    g.add_node("assistant", call_model)
    g.add_edge(START, "assistant")
    g.add_edge("assistant", END)
    return g.compile()

graph = construct_graph()

if __name__ == "__main__":
    example_order = {"order_id":"A12345"}
    convo = [HumanMessage(content="Por favor, cancele meu pedido A12345.")]
    result = graph.invoke({"order":example_order, "messages":convo})
    for msg in result["messages"]:
        print(f"{msg.type}: {msg.content}")


