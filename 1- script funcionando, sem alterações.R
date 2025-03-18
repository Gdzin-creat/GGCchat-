

#----------------------------------configurações iniciais e instalação de pacotes
install.packages("reticulate")

#================================ leitura das bibliotecas
library(reticulate)
library(stringr)
#-----------------------------------criação do ambiente 
virtualenv_create(
  envname = "langchain_rag",
  packages = c(
    "langchain",
    "langchain-community",
    "pypdf",
    "pinecone",
    "langchain_pinecone",
    "langchain-openai",
    "pinecone-client[grpc]"
  )
)
reticulate::use_virtualenv("langchain_rag")
reticulate::py_config()

#===========================teste para ver se o ambiente está corretamente configurado
reticulate::py_run_string("print('Ambiente configurado!')")

#================================ Alterando o valor da variável "OPENAI_API_KEY" e carregamentos das chaves 


chave_api_openai <- Sys.getenv("OPENAI_API_KEY")
chave_api_pinecone <- Sys.getenv("PINECONE_API_KEY")
file.edit("~/.Renviron")

#========================================== Criar o diretório, se não existir
if (!dir.exists("docs")) {
  dir.create("docs")
}

# Fazer o download novamente
download.file(
  "https://cran.r-project.org/web/packages/ggplot2/ggplot2.pdf",
  destfile = "docs/ggplot2.pdf",
  mode = "wb"
)

#leitura do pdf

#================================================= Código Python incorporado no R
reticulate::py_run_string("
from langchain_community.document_loaders import PyPDFLoader

# Carregar o arquivo PDF
loader = PyPDFLoader('docs/ggplot2.pdf')

# Processar as páginas do PDF
paginas = loader.load()

# Retornar o tipo como string
tipo_paginas = str(type(paginas))
")

# Verificar o tipo e o número de páginas no R
print(reticulate::py_run_string("print(type(paginas))")$output)
print(py$tipo_paginas)
print(length(py$paginas))

# Acessar os objetos Python no R
paginas_type <- py$paginas
cat('Tipo do objeto páginas:', reticulate::py_run_string("print(type(paginas))")$output, "\n")
cat('Número de páginas:', length(paginas_type), "\n")


#utilização do R para puxar informações do python
paginas_em_r <- py$paginas

# metadatos da primeira pagina
paginas_em_r[[1]]$metadata 

# quantidade de caracteres da centésima pagina
nchar(paginas_em_r[[100]]$page_content) 

# Executar o código Python no R
reticulate::py_run_string("
from langchain.text_splitter import CharacterTextSplitter

# Configurar os tamanhos para o divisor
tamanho_pedaco = 4000
tamanho_intersecao = 150

# Criar o divisor de documentos
divisor_documentos = CharacterTextSplitter(
  chunk_size = tamanho_pedaco, 
  chunk_overlap = tamanho_intersecao, 
  separator = ' '
)

# Dividir as páginas do PDF em partes
partes_pdf = divisor_documentos.split_documents(paginas)
")

# Acessar o número de partes geradas
numero_partes <- length(py$partes_pdf)
cat("Número de partes:", numero_partes, "\n")

# Opcional: Inspecionar as partes do PDF no R
partes_pdf <- py$partes_pdf
print(partes_pdf)

#calculando o custo que o pdf nos da por token;  US$ 0,10 por 1 milhão de tokens
total_tokens <- purrr::map_int(partes_pdf, 
                               ~ TheOpenAIR::count_tokens(.x$page_content)) |> 
  sum()
print(total_tokens)
################################### tudo certo daqui pra cima



# Define as variáveis do ambiente R para o Python
reticulate::py_install("langchain_openai", pip = TRUE)



py_run_string('
import os
os.environ["OPENAI_API_KEY"] = r.chave_api_openai
os.environ["PINECONE_API_KEY"] = r.chave_api_pinecone
')

# Importa o módulo langchain_openai
langchain_openai <- import("langchain_openai")

# Configura o modelo de embeddings
OpenAIEmbeddings <- langchain_openai$OpenAIEmbeddings
embeddings <- OpenAIEmbeddings(model = "text-embedding-ada-002")

# Supondo que partes_pdf[100]$page_content seja equivalente ao conteúdo de um PDF
# Certifique-se de que partes_pdf está definido corretamente em R
resultado <- embeddings$embed_query(partes_pdf[[100]]$page_content)

# Imprime o resultado
print(resultado)

#=========================== ligar agora o pinecone para trabalhar

# Código atualizado para R usando reticulate
reticulate::py_run_string("
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os

# Inicializar o cliente Pinecone
pc = Pinecone(os.environ['PINECONE_API_KEY'])

# Nome do índice
nome_indice = 'rag-ggplot'

# Criar índice se não existiri
if nome_indice not in pc.list_indexes().names():
    pc.create_index(
        name = nome_indice,
        dimension = 1536,
        metric = 'cosine',
        spec = ServerlessSpec(
            cloud = 'aws', 
            region = 'us-east-1'
        )
    )

# Carregar embeddings com a nova importação
embeddings = OpenAIEmbeddings()

# Configurar a base de conhecimento
base_conhecimento = PineconeVectorStore.from_existing_index(
    index_name=nome_indice,
    embedding=embeddings
)

# Criar a loja de vetores a partir dos documentos
vetores_dados = PineconeVectorStore.from_documents(
    partes_pdf,
    index_name=nome_indice,
    embedding=embeddings
)

# Pergunta e busca de similaridade
pergunta = 'Como rotacionar o texto no eixo x de um gráfico ggplot?'
resultado = base_conhecimento.similarity_search(pergunta, k=3)
")

reticulate::py_run_string("
if resultado:
    resposta = resultado[0]  # Pegar o primeiro resultado, se existir
else:
    resposta = 'Nenhum resultado encontrado.'
")

# Converte o objeto Python em um valor utilizável no R
resposta_python <- py$resposta

# Verifica se a resposta é um objeto Python com estrutura complexa, e obtém o texto
resposta_texto <- as.character(resposta_python)

# Agora imprime a resposta corretamente
cat("Resposta:", resposta_texto, "\n")


# Configurar modelo OpenAI e carregar o índice Pinecone de forma organizada
reticulate::py_run_string("
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Configurar modelo OpenAI
llm = ChatOpenAI(
    openai_api_key = os.environ['OPENAI_API_KEY'],
    model_name = 'gpt-4o-mini',
    temperature = 0.0
)

# Nome do índice e configuração dos embeddings
nome_indice = 'rag-ggplot'
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# Carregar a base de conhecimento do índice existente
base_conhecimento = PineconeVectorStore.from_existing_index(
    index_name=nome_indice,
    embedding=embeddings
)

# Criar a cadeia RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=base_conhecimento.as_retriever()
)

# Pergunta usando invoke
pergunta = 'Como fazer gráfico de linha no ggplot?'
resposta = qa.invoke({'query': pergunta})
")

# Recuperar a resposta em R
resposta <- py$resposta
# Exibir a resposta convertida em texto
cat("Resposta:", paste(resposta, collapse = " "), "\n")

#==================== criação de um chatbot que guarde informações de sua pergunta

# Criar o script Python em R
py_run_string("
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever

# Definir o modelo LLM e o recuperador
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
recuperador = base_conhecimento.as_retriever()
")


# Criar o recuperador de contexto
# Criar o modelo LLM
py_run_string("
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_openai import ChatOpenAI

# Criar o modelo LLM
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
")

# Criar o prompt de contextualização
py_run_string("
prompt_contextualizacao = (
    'Dado um histórico de chat e a última pergunta do usuário '
    'que pode referenciar o contexto no histórico do chat, '
    'formule uma pergunta independente que possa ser compreendida '
    'sem o histórico do chat. NÃO responda à pergunta, '
    'apenas reformule-a se necessário e, caso contrário, retorne-a como está.'
)

prompt_template = ChatPromptTemplate.from_messages([
    ('system', prompt_contextualizacao),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
])
")

# Criar o recuperador de contexto
py_run_string("
recuperador_historico = create_history_aware_retriever(
    llm=llm, retriever=recuperador, prompt=prompt_template
)
")

# Criar o prompt de perguntas e respostas
py_run_string("
prompt_final = (
    'Você é um assistente para tarefas de perguntas e respostas sobre o ggplot2. '
    'Use os seguintes trechos de contexto recuperado para responder '
    'à pergunta. Se você não souber a resposta, diga que você '
    'não sabe. Se a pergunta estiver fora do contexto recuperado, '
    'não responda e apenas diga que está fora do contexto.'
    'A sua resposta deve ser em português brasileiro.\\n\\n'
    '{context}'
)

qa_prompt = ChatPromptTemplate.from_messages([
    ('system', prompt_final),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
])
")

# Criar a cadeia de perguntas e respostas
py_run_string("
cadeia_perguntas_e_respostas = create_stuff_documents_chain(
    llm, 
    qa_prompt
)
")

# Criar a cadeia RAG (Retrieval-Augmented Generation)
py_run_string("
cadeia_rag = create_retrieval_chain(
    recuperador_historico, 
    cadeia_perguntas_e_respostas
)
")
py_run_string("
sessoes = {}

def obter_historico_sessao(id_sessao: str):
    if id_sessao not in sessoes:
        from langchain_community.chat_message_histories import ChatMessageHistory
        sessoes[id_sessao] = ChatMessageHistory()
    return sessoes[id_sessao]
")

# Criar o objeto chat_cadeia_rag no ambiente Python
py_run_string("
from langchain_core.runnables.history import RunnableWithMessageHistory

chat_cadeia_rag = RunnableWithMessageHistory(
    cadeia_rag,
    obter_historico_sessao,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)

def obter_resposta(pergunta, id_sessao='abc123'):
    resultado = chat_cadeia_rag.invoke(
        {'input': pergunta},
        config={'configurable': {'session_id': id_sessao}}
    )
    return resultado
")

library(reticulate)
resposta <- py$obter_resposta("Qual a capital do Ceará?")
print(resposta$answer)

iniciar_chat <- function(id_sessao = "abc123") {
  print("Iniciando chatGGPLOT, digite 'sair' para terminar.")
  
  while(TRUE){
    pergunta <- readline("Diga: ")
    
    if (stringr::str_to_lower(pergunta) == 'sair'){
      print("Encerrando o chatGGPLOT.")
      break
    }
    
    resultado <- py$obter_resposta(pergunta, id_sessao)
    print(resultado$answer)
  }
}

iniciar_chat()


 

