import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_core.runnables import RunnablePassthrough

# from functions.pinecone_functions import insert_or_fetch_embeddings

new_template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 

{context} 

Question: {question} 

Helpful Answer: 
"""


def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retrieving ... ")
    print("-" * 200)

    query = "What is Pinecone in machine learning?"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    vector_store = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    # retrieval_qa_chat_prompt = hub.pull(
    #     "langchain-ai/retrieval-qa-chat"
    # )  # Prompt to send to the llm after retrieving information (context based on embedded chunks/split text); https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat?organizationId=c01486a8-6560-59ec-8081-6bf36f571f20

    new_template = new_template

    custom_rag_prompt = PromptTemplate.from_template(template=new_template)

    rag_chain = (
        {
            "context": vector_store.as_retriever() | format_documents,
            "question": RunnablePassthrough(),
        }
        | custom_rag_prompt
        | llm
    )

    response = rag_chain.invoke(
        query
    )  # Here, because we are not using langchain to create the chain, we must use rag_chain.invoke(query), not rag_chain.invoke(input={"input": query})

    print(response)
    print("-" * 100)
    print(response.content)
