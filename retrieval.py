import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# from functions.pinecone_functions import insert_or_fetch_embeddings

if __name__ == "__main__":
    print("Retrieving ... ")
    print("-" * 200)

    query = "What is Pinecone in machine learning?"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    vector_store = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull(
        "langchain-ai/retrieval-qa-chat"
    )  # Prompt to send to the llm after retrieving information (context based on embedded chunks/split text); https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat?organizationId=c01486a8-6560-59ec-8081-6bf36f571f20

    print(retrieval_qa_chat_prompt)
    print("-" * 100)
    print(type(retrieval_qa_chat_prompt))
    print("-" * 100)
    print(retrieval_qa_chat_prompt.messages)
    print("-" * 100)

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
     
    print(combine_docs_chain)
    print("-"*100)

    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": query})

    print(result)
    print("-" * 50)
    print(type(result))
    print("-" * 50)
    print(result["input"])
    print("-" * 50)
    print(result.keys())
    print("-" * 50)
    print(result["answer"])
    
    # Used LangSmith to view what is happening when retrieving context, formatting documents/prompt, and calling the llm.
