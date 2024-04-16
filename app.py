import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def process_input(urls, question):

    llm = Ollama(model="llama2")
    # List of urls input by the user seperated by new line
    urls_list = urls.split("\n")
    # loads docs for each url input by the user
    docs = [WebBaseLoader(url).load() for url in urls_list]
    # list containing html text of all docs accessed using url
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 7500, chunk_overlap = 100)
    texts = text_splitter.split_documents(docs_list)

    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device': 'cpu'})

    vectorstore = FAISS.from_documents(texts, embeddings)

    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """

    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])

    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       #search_kwargs = 2 returns top 2 responses based on input query
                                       retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       #injects our custom prompt into the chain
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    
    response = qa_chain({'query':question})
    print(response['source_documents'])

    return response['result']

# Streamlit code

st.set_page_config(layout='centered', page_title='Query your URLs')
st.header("URL Querying using Ollama🦙")
st.write("Enter URLs (one per line) and a question to query the URLs.")

url = st.text_area("Enter your URLs seperated by a new line")

user_input = st.chat_input("Ask your question...")

if user_input:
    with st.spinner("Generating response..."):
        answer = process_input(url, user_input)
        st.success(answer)