from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import uuid
import os

load_dotenv()

app = FastAPI()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
api_key = os.getenv('GROQ_API_KEY')

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
llm = ChatGroq(groq_api_key=api_key, model_name='llama3-8b-8192')

session_store = {}

def load_documents():
    pdf_paths = [
        "rp/Govt Helpline Nos.pdf",
        "rp/Personal Safety.pdf",
        "rp/Safety_of_street_The_role_of_street_design.pdf",
        "rp/Sanraksha.pdf"
    ]
    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        documents.extend(docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

splits = load_documents()
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate([
    ('system', contextualize_q_system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

system_prompt = (
    "You are a safety expert 'Kavach' assisting users with their safety-related concerns "
    "and providing guidance on Sanraksha app features. Use the retrieved context from safety guidelines to provide CONCISE answers. "
    "If you don't have relevant information, say 'I am not sure' rather than making up an answer."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate([
    ('system', system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human', '{input} in short')
])

ques_ans_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, ques_ans_chain)

@app.middleware("http")
async def session_middleware(request: Request, call_next):
    session_id = request.cookies.get("session_id")
    
    if not session_id or request.url.path == "/":
        session_id = str(uuid.uuid4())
        session_store[session_id] = ChatMessageHistory()
    
    request.state.session_id = session_id
    response = await call_next(request)
    
    if request.url.path == "/":
        response.set_cookie(key="session_id", value=session_id)
    
    return response

def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key='answer'
)

@app.post("/ask/")
async def ask_question(request: Request, question: dict):
    session_id = request.state.session_id
    session_history = get_session_history(session_id)

    user_input = question.get("input")
    if not user_input:
        raise HTTPException(status_code=400, detail="No input provided")

    response = conversational_rag_chain.invoke(
        {'input': user_input},
        config={'configurable': {'session_id': session_id}}
    )

    return JSONResponse(content={"answer": response['answer']})

