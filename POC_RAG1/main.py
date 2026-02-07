from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from dotenv import load_dotenv

from service_2 import criar_evento

load_dotenv()

CAMINHO_DB = 'db'

class DecisaoAgendamento(BaseModel):
    marcar: bool = Field(description="True se o usuário quer marcar consulta")
    medico: str | None = Field(description="Nome do médico, se houver")
    horario: str | None = Field(description="Horário hh:mm, se houver")
    explicacao: str = Field(description="Resumo curto da decisão")


prompt_template = """


Você é um agente especializado em marcar consultas de agenda no google agendas.
Responda a pergunta que o usuário passar e se apresente como Ophélia: {pergunta}
Utilize a seguinte base de conhecimento para responder a pergunta:
{base_conhecimento}

"""

def perguntar():
    interruptor = False
    A = []
    while not interruptor and len(A) < 5:
        #if len(A) >= 1:
            # prompt_template = prompt_template.replace("e se apresente como Ophélia", "")

        pergunta_usuario = input("Chat conversasional:")
        A.append(pergunta_usuario)
        print(len(A))

        # carrega do banco de dados
        funcao_enbedding = OpenAIEmbeddings()
        db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_enbedding)

        # Comparar a pergunta do usuário com o banco vetorizado
        resultados = db.similarity_search_with_relevance_scores(pergunta_usuario, k=5)
        if len(resultados) == 0:
            print("Nenhum resultado foi encontrado.")
            return
        texto_resultados = []
        for resultado in resultados:
            texto_resultados.append(resultado[0].page_content)

        base_conhecimento = "\n\n----\n\n".join(texto_resultados)
        prompt = ChatPromptTemplate.from_template(prompt_template)
        prompt = prompt.invoke({"base_conhecimento" : base_conhecimento, "pergunta" : pergunta_usuario})

        modelo = ChatOpenAI(model="gpt-5.2", temperature=0)
        texto_resposta = modelo.invoke(prompt).content
        print(texto_resposta)

        prompt_template_decisao = """Me responda com base na string passada pelo usuário: {entrada_2} e na resposta dada pelo agente de IA: {entrada_IA} se o usuário está pedindo para marcar 
        algum horário com algum médico e se este médico tem o horário disponível. Só me retorne positivo se você encontrar o nome do médico e o horário do agendamento Se sim: escreva: SIM, o usuário quer ... no horário hh:mm com o doutor NOME.  Se não, escreva NÃO, o usuário não quer... Sempre responda
        dessas duas maneiras, mesmo se não entender a pergunta. Responda APENAS no formato JSON seguido este schema: {schema}
        
        
        """
        prompt_decisao = ChatPromptTemplate.from_template(prompt_template_decisao)
        entrada2 = "\n---\n".join(A)
        '''
        prompt_decisao = prompt_decisao.invoke(prompt_decisao.format(
            entrada_2=entrada2,
            schema=DecisaoAgendamento.model_json_schema()
        ))'''
        modelo2 = ChatOpenAI(model="gpt-5.2", temperature=0).with_structured_output(DecisaoAgendamento)
        resposta_decisão = modelo2.invoke(prompt_decisao.format(
            entrada_2=entrada2,
            entrada_IA=texto_resposta,
            schema=DecisaoAgendamento.model_json_schema()
        ))


        # Verifica ativação de marcar horário
        #ativação_horario = resposta_decisão.__contains__("SIM")
        #print("Contem" if ativação_horario else "Não contém")

        medico = resposta_decisão.medico
        horario = resposta_decisão.horario
        marcar = resposta_decisão.marcar

        print(resposta_decisão.marcar)
        print(resposta_decisão.medico)
        print(resposta_decisão.horario)


        if marcar:
            criar_evento(summary=medico, time=horario + ":00")
            print("Horário marcado com sucesso")

    return

perguntar()









