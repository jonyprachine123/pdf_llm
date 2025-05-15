import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['VECTOR_STORE_PATH'] = 'vector_store'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'txt', 'md'}

# Create necessary folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VECTOR_STORE_PATH'], exist_ok=True)

# Global variables to store conversation chain, chat history, and current document info
conversation_chain = None
chat_history = []
current_document_info = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_document(file_path):
    # Load document based on file extension
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
    elif file_extension == 'docx':
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
    elif file_extension in ['txt', 'md']:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
        
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create a unique identifier for this document
    import hashlib
    doc_hash = hashlib.md5(file_path.encode()).hexdigest()
    vector_store_dir = os.path.join(app.config['VECTOR_STORE_PATH'], doc_hash)
    
    # Create vector store and save it to disk
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(vector_store_dir)
    
    # Store document info globally
    global current_document_info
    current_document_info = {
        'file_path': file_path,
        'vector_store_dir': vector_store_dir,
        'doc_hash': doc_hash,
        'chunks_count': len(chunks)
    }
    
    # Create conversation chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
    
    global conversation_chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    
    return len(chunks)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Check if we already have processed this file before
            import hashlib
            doc_hash = hashlib.md5(file_path.encode()).hexdigest()
            vector_store_dir = os.path.join(app.config['VECTOR_STORE_PATH'], doc_hash)
            
            # If vector store exists, load it instead of reprocessing
            if os.path.exists(vector_store_dir):
                # Load existing vector store
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.load_local(vector_store_dir, embeddings)
                
                # Create conversation chain
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
                
                global conversation_chain, current_document_info
                conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vector_store.as_retriever(),
                    memory=memory
                )
                
                # Update document info
                current_document_info = {
                    'file_path': file_path,
                    'vector_store_dir': vector_store_dir,
                    'doc_hash': doc_hash,
                    'chunks_count': 'unknown (loaded from cache)'
                }
                
                return jsonify({
                    'message': f'File loaded from cache. Ready for questions.',
                    'filename': filename,
                    'cached': True
                })
            else:
                # Process the document normally
                chunks_count = process_document(file_path)
                return jsonify({
                    'message': f'File processed successfully. Created {chunks_count} chunks.',
                    'filename': filename,
                    'cached': False
                })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    global conversation_chain, chat_history
    
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    if not conversation_chain:
        return jsonify({'error': 'Please upload a PDF first'}), 400
    
    try:
        response = conversation_chain.invoke({"question": question})
        answer = response.get('answer')
        chat_history.append((question, answer))
        
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    global conversation_chain, chat_history, current_document_info
    conversation_chain = None
    chat_history = []
    
    # Clear uploads folder
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # Don't delete vector store to keep it persistent
    # Just reset the current document info
    current_document_info = None
    
    return jsonify({'message': 'Conversation reset successfully'})

if __name__ == '__main__':
    app.run(debug=True)
