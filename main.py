from fastapi import FastAPI, File, UploadFile, HTTPException
import fitz
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Define global objects for storing resources
class GlobalResources:
    def __init__(self):
        self.vectorstore = None
        self.conversation = None

g = GlobalResources()

# Define helper functions

async def extract_text_from_pdf(upload_file: UploadFile) -> str:
    pdf_bytes = await upload_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc: 
        text += page.get_text()
    return text

def get_text_chunks(text): 
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks): 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = GoogleGenerativeAI(model="gemini-pro")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def generate_answer(user_question, conversation):
    # Generate answer using the conversation chain
    response = conversation.invoke(query=user_question)
    answer = response.output

    return answer

def handle_user_input(user_question):
    try:
        # Check if the conversation chain exists
        if not hasattr(g, "conversation") or g.conversation is None:
            raise Exception("Conversation chain not found")

        # Retrieve conversation chain
        conversation = g.conversation

        # Generate answer using the conversation chain
        answer = generate_answer(user_question, conversation)

        # Print or log the answer
        print(f"User: {user_question}")
        print(f"Bot: {answer}")

        return answer

    except Exception as e:
        print(f"Error handling user input: {str(e)}")
        return None

# Define routes

# upload pdf route
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("application/pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a PDF")
    
    try:
        # get pdf text
        get_pdf_text = await extract_text_from_pdf(file)
        # get the text chunks 
        text_chunks = get_text_chunks(get_pdf_text) 

        # create vector store for embeddings
        vectorstore = get_vectorstore(text_chunks)

        # create converstion chain
        conversation = get_conversation_chain(vectorstore)

        # Store resources for later use
        g.vectorstore = vectorstore
        g.conversation = conversation

        return {"details": "PDF upload successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to process PDF: " + str(e))
    
# process question route

# route for chatting with the pdf
@app.post("/process/")
async def process_question(question_payload: dict):
    try:
        # Extract question from payload
        question = question_payload["question_payload"]["question"]
        
        # Extract conversation history from payload
        conversation_history = question_payload.get("conversation_history", [])

        # Ensure conversation history is initialized correctly
        if not conversation_history:
            conversation_history = []

        # Check for uploaded PDF
        if not g.vectorstore:
            return {"error": "Please upload a PDF first using the /upload/ endpoint"}

        # Handle user input
        answer = handle_user_input(question)

        # Update conversation history
        conversation_history.append({"question": question, "answer": answer})

        return {"answer": answer}
    except Exception as e:
        return {"error": f"Failed to process question: {str(e)}"}
