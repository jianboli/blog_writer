# importing the modules
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.agents import Tool
from langchain.agents import initialize_agent

# defining the model
llm = ChatOpenAI(
    openai_api_key="",
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

# loading the document
loader = PyPDFLoader("./3DPrinter_Manual.pdf")
mypdf = loader.load() 

# Defining the splitter 
document_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 70
)

# splitting the document
docs = document_splitter.split_documents(mypdf)

# embedding the chunks to vector stores
embeddings = OpenAIEmbeddings(openai_api_key="")
persist_directory = 'db'

my_database = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

# defining the conversational memory
retaining_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

# defining the retriever
question_answering = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=my_database.as_retriever(),
    memory=retaining_memory
)

# defining the tool for the agent
tools = [
    Tool(
        name='Knowledge Base',
        func=question_answering.run,
        description=(
            'use this tool when answering questions related to the 3D printer'
        )
    )
]

# initializing the agent
agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=retaining_memory
)

# calling the agent
while True:
    question = input("Enter your query: ")
    if question == 'exit': 
	    break 
    print(agent(question))


