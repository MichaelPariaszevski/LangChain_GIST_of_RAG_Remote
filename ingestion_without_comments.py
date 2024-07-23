from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter

load_dotenv(find_dotenv(), override=True)

# import os

if __name__ == "__main__":
    print("Ingesting ... ")
    print("-" * 200)

    loader = TextLoader(
        "medium_blog.txt", encoding="UTF-8"
    )  # Here, encoding="UTF-8" is necessary

    # This loads the file into a LangChain Document (LangChain doucment object)
    document = loader.load()

    print("splitting ... ")
    print("-" * 200)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    texts = text_splitter.split_documents(document)

    print(f"Created {len(texts)} chunks")
    print("-" * 200)
    print("Ingesting, Embedding, and Storing to Pinecone ... ")
    print("-" * 200)

    # from functions.pinecone_functions import delete_pinecone_index
    
    from functions.pinecone_functions import insert_or_fetch_embeddings

    insert_or_fetch_embeddings(
        index_name="langchain-gist-of-rag", chunks=texts, delete=False
    )

    print("Finished")
    

