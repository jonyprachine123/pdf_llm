# PDF LLM Reader

A Flask-based application that allows users to upload PDF documents, process them using LangChain and a vector database (FAISS), and then ask questions about the content using Google's Gemini API.

## Features

- PDF document upload and processing
- Text extraction and chunking for efficient retrieval
- Vector embeddings using Google's Embedding API
- Conversational interface with the Gemini LLM
- Persistent conversation history
- ChatGPT-style UI with dark mode support
- Typewriter effect for bot responses
- Mobile-responsive design

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key

## Installation

1. Clone the repository:
```
git clone <repository-url>
cd pdf_llm
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Set up your environment variables:
   - Create a `.env` file in the root directory
   - Add your Google API key: `GOOGLE_API_KEY=your_api_key_here`

## Usage

1. Start the Flask application:
```
python app.py
```

2. Open your web browser and navigate to `http://127.0.0.1:5000`

3. Upload a PDF document using the form

4. Once the PDF is processed, you can start asking questions about its content

5. Use the "Reset Conversation" button to clear the conversation history and delete the uploaded PDF

## How It Works

1. **PDF Processing**: When a PDF is uploaded, it's processed using PyPDF to extract text.

2. **Text Chunking**: The extracted text is split into manageable chunks using LangChain's RecursiveCharacterTextSplitter.

3. **Vector Embeddings**: Each chunk is converted into vector embeddings using Google's Embedding API.

4. **Vector Database**: The embeddings are stored in a FAISS vector database for efficient similarity search.

5. **Question Answering**: When a user asks a question, the system:
   - Converts the question to an embedding
   - Finds the most relevant chunks in the vector database
   - Sends the question and relevant context to the Gemini LLM
   - Returns the generated answer to the user

## License

[MIT License](LICENSE)
