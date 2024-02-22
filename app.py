import os
from flask import Flask, request, jsonify
import textwrap
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
from flask_cors import CORS

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""  # change this

app = Flask(__name__)
CORS(app)

loader = TextLoader(r"C:\Dhanush\Angular-CRUD\Backend\data.txt")
  # change this
document = loader.load()

def text_wrap_preserves_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(document)
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 1.0, "max_length": 800})
chain = load_qa_chain(llm, chain_type="stuff")

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        query_text = data.get('query', '')
        result = process_query(query_text)
        response = {"result": result}
        return jsonify(response)
    except Exception as e:
        error_response = {"error": str(e)}
        return jsonify(error_response), 400

def process_query(query_text):
    if query_text.lower() == 'exit':
        return "Exiting..."
    if query_text.strip().lower() in ["hi", "hello", "hey"]:
        return "Hello! How can I assist you today?"
    if query_text.strip().lower() in ["bye"]:
        return "Goodbye! Have a great day!"
    
    docs_result = db.similarity_search(query_text)
    if docs_result:
        # Run question-answering chain on the most similar document
        response = chain.run(input_documents=docs_result, question=query_text)
        return text_wrap_preserves_newlines(response)
    else:
        return "I'm sorry, the question is not related to the document or no information was found."
            #its not working for higher threshold i think we need to create some similarity functions .
            #similarity_threshold = -0.5
            #top_result_score = docs_result[0].metadata.get("faiss_score", 0)

            #if top_result_score >= similarity_threshold:
               # response = chain.run(input_documents=docs_result, question=query_text)
               # return text_wrap_preserves_newlines(response)
            #else:
                #return "I'm sorry, the question is not closely related to the document or no relevant information was found."
       # else:
           # return "I'm sorry, the question is not related to the document or no information was found."

if __name__ == "__main__":
    app.run(debug=True)