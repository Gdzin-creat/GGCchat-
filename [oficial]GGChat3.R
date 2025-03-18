#install.packages("reticulate")
#install.packages("markdown")
#================================ Leitura das bibliotecas

library(reticulate)
library(shiny)
library(markdown)
#----------------------------------- Criação do ambiente virtual
virtualenv_create(
  envname = "langchain_rag",
  packages = c(
    "langchain",
    "langchain-community",
    "pypdf",
    "pinecone-client[grpc]",
    "langchain_pinecone",
    "langchain-openai"
  )
)
reticulate::use_virtualenv("langchain_rag")
reticulate::py_config()

#=========================== Configuração das APIs

chave_api_openai <- Sys.getenv("OPENAI_API_KEY")
chave_api_pinecone <- Sys.getenv("PINECONE_API_KEY")

reticulate::py_run_string("
import os
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 🔹 Configurar credenciais
os.environ['OPENAI_API_KEY'] = r.chave_api_openai
os.environ['PINECONE_API_KEY'] = r.chave_api_pinecone

# 🔹 Inicializar cliente Pinecone
pc = Pinecone(os.environ['PINECONE_API_KEY'])
nome_indice = 'meus-documentos'

# 🔹 Criar índice, se não existir
if nome_indice not in pc.list_indexes().names():
    pc.create_index(
        name=nome_indice,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# 🔹 Configurar embeddings e conectar ao Pinecone
embeddings = OpenAIEmbeddings()
base_conhecimento = PineconeVectorStore.from_existing_index(
    index_name=nome_indice,
    embedding=embeddings
)

# 🔹 Configurar modelo OpenAI para responder apenas com base no PDF
llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.0)

# 🔹 Criar retriever restritivo para buscar no Pinecone
retriever = base_conhecimento.as_retriever(search_kwargs={'k': 3})

# 🔹 Criar prompt para reformulação de perguntas
prompt_contextualizacao = ChatPromptTemplate.from_messages([
    ('system', \"\"\"Dado um histórico de chat e a última pergunta do usuário, 
               reformule a pergunta para que possa ser compreendida isoladamente.\"\"\"),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
])

# 🔹 Criar recuperador de histórico
recuperador_historico = create_history_aware_retriever(
    llm=llm, retriever=retriever, prompt=prompt_contextualizacao
)

# 🔹 Criar prompt para perguntas e respostas (melhorar !!!!)
prompt_final = ChatPromptTemplate.from_messages([
    ('system', \"\"\"Você é um assistente para perguntas e respostas sobre PDFs enviados pelo usuário. 
               Use os seguintes trechos de contexto recuperado para responder à pergunta. 
               Se você não souber a resposta, diga que você não sabe. 
               Se a pergunta estiver fora do contexto recuperado, apenas informe isso. 
               A sua resposta deve ser em português brasileiro.\n\n{context}\"\"\"),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
])

# 🔹 Criar a cadeia de perguntas e respostas
cadeia_perguntas_e_respostas = create_stuff_documents_chain(
    llm, 
    prompt_final
)

# 🔹 Criar a cadeia RAG (Retrieval-Augmented Generation)
cadeia_rag = create_retrieval_chain(
    recuperador_historico, 
    cadeia_perguntas_e_respostas
)

# 🔹 Criar histórico de sessões com limite de mensagens
sessoes = {}

def obter_historico_sessao(id_sessao: str, limite=20):
    if id_sessao not in sessoes:
        sessoes[id_sessao] = ChatMessageHistory()
    historico = sessoes[id_sessao]
    
    while len(historico.messages) > limite:
        historico.messages.pop(0)
    
    return historico

# 🔹 Criar objeto chat_cadeia_rag
chat_cadeia_rag = RunnableWithMessageHistory(
    cadeia_rag,
    obter_historico_sessao,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)

# 🔹 Função para adicionar PDFs ao banco de dados
def adicionar_pdf(caminho_pdf):
    loader = PyPDFLoader(caminho_pdf)
    paginas = loader.load()
    base_conhecimento.add_documents(paginas)
    return 'PDF adicionado com sucesso.'

# 🔹 Função para perguntar com histórico de contexto
def perguntar(pergunta, id_sessao='abc123'):
    resultado = chat_cadeia_rag.invoke(
        {'input': pergunta},
        config={'configurable': {'session_id': id_sessao}}
    )
    return resultado
")
#interface 


ui <- fluidPage(
  titlePanel("Resenha com GGChat"),
  sidebarLayout(
    sidebarPanel(
      fileInput("pdf_file", "Envie um arquivo PDF", accept = ".pdf"),
      actionButton("add_pdf", "Adicionar PDF"),
      textInput("question", "Digite sua pergunta:"),
      actionButton("ask", "Perguntar")
    ),
    mainPanel(
      uiOutput("answer")  # Exibe o histórico formatado
    )
  )
)

server <- function(input, output, session) {
  conversa <- reactiveVal("")  # 🔹 Armazena o histórico da conversa
  session_id <- reactiveVal(paste0("sessao_", session$token))  # 🔹 ID de sessão único para cada usuário
  
  observeEvent(input$add_pdf, {
    req(input$pdf_file)
    caminho_pdf <- input$pdf_file$datapath
    print(paste("Caminho do PDF recebido:", caminho_pdf))
    reticulate::py$adicionar_pdf(caminho_pdf)
  })
  
  observeEvent(input$ask, {
    req(input$question)
    
    # 🔹 Obtendo a resposta do Python com histórico de chat
    resposta <- reticulate::py$perguntar(input$question, id_sessao = session_id())
    
    # 🔹 Converter resposta para string, se necessário
    if (is.list(resposta)) {
      resposta <- paste(unlist(resposta), collapse = " ")
    }
    
    # 🔹 Atualizar o histórico da conversa
    conversa_atual <- conversa()
    nova_mensagem <- paste0("**Você:** ", input$question, "\n\n**GGChat:** ", resposta, "\n\n")
    conversa(paste(conversa_atual, nova_mensagem, sep = "\n"))
  })
  
  # 🔹 Exibir o histórico completo da conversa
  output$answer <- renderUI({
    HTML(markdown::markdownToHTML(text = conversa(), fragment.only = TRUE))
  })
}

shinyApp(ui, server)

