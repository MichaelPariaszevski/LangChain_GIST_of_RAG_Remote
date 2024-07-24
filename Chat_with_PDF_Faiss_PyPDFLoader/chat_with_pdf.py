import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import CharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_community.vectorstores import FAISS

from langchain.chains.retrieval import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain import hub

if __name__ == "__main__":
    print("Hello")

    pdf_file_path = (
        "Chat_with_PDF_Faiss_PyPDFLoader/React_synergizing_reasoning_and_acting.pdf"
    )

    loader = PyPDFLoader(file_path=pdf_file_path)

    documents = loader.load()  # documents is made up of LangChain document objects

    print(documents)
    print("-" * 100)

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )

    documents_chunked = text_splitter.split_documents(documents=documents)

    print(documents_chunked)
    print("-" * 100)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = FAISS.from_documents(
        documents=documents_chunked, embedding=embeddings
    )

    vector_store.save_local("faiss_index_react")

    new_vector_store = FAISS.load_local(
        folder_path="faiss_index_react",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    ) # Susceptible to a deserialization attack
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    retrieval_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat") 
    
    combine_docs_chain=create_stuff_documents_chain(llm, retrieval_qa_chat_prompt) 
    
    retrieval_chain = create_retrieval_chain(
            retriever=vector_store.as_retriever(), combine_docs_chain=combine_docs_chain
        )
    
    query="Give me the gist of ReAct in 3 sentences"
    
    response=retrieval_chain.invoke(input={"input": query}) 
    
    print(response)
    print("-"*100) 
    print(response["answer"])