from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import TextLoader

# Text Document Loader Documentation

# """Load text files."""
# from typing import List

# from langchain.docstore.document import Document
# from langchain.document_loaders.base import BaseLoader


# class TextLoader(BaseLoader):
#     """Load text files."""

#     def __init__(self, file_path: str):
#         """Initialize with file path."""
#         self.file_path = file_path

#     def load(self) -> List[Document]:
#         """Load from file path."""
#         with open(self.file_path) as f:
#             text = f.read()
#         metadata = {"source": self.file_path}
#         return [Document(page_content=text, metadata=metadata)]


from langchain_text_splitters import CharacterTextSplitter

# from langchain_pinecone import PineconeVectorStore # Not being used for now

load_dotenv(find_dotenv(), override=True)

import os

if __name__ == "__main__":
    print("Ingesting ... ")
    print("-" * 200)

    # import chardet

    # def detect_encoding(file_path):
    #     with open(file_path, 'rb') as f:
    #         result = chardet.detect(f.read())
    #     return result['encoding']

    # encoding = detect_encoding("medium_blog.txt")

    # with open("medium_blog.txt", 'r', encoding=encoding) as f:
    #     text = f.read()

    # with open("medium_blog_utf8.txt", 'w', encoding='utf-8') as f:
    #     f.write(text)

    loader = TextLoader(
        "medium_blog.txt", encoding="UTF-8"
    )  # Here, encoding="UTF-8" is necessary

    # This loads the file into a LangChain Document (LangChain doucment object)
    document = loader.load()

    print("splitting ... ")
    print("-" * 200)
    # print(document[0].page_content)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    texts = text_splitter.split_documents(document)

    print(f"Created {len(texts)} chunks")
    print("-" * 200)
    print("Ingesting, Embedding, and Storing to Pinecone ... ")

    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    from functions.pinecone_functions import delete_pinecone_index

    def insert_or_fetch_embeddings(index_name: str, chunks, delete=False):
        import pinecone
        from langchain_community.vectorstores import Pinecone
        from langchain_openai import OpenAIEmbeddings
        from pinecone import PodSpec

        pc = pinecone.Pinecone()
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

        print(pc.list_indexes())
        print("-" * 50)
        print(pc.list_indexes().names())
        print("-" * 50)

        if index_name == "all":
            delete_pinecone_index(index_name="all")
            return None

        elif (
            index_name != "all"
            and index_name in pc.list_indexes().names()
            and delete == True
        ):
            delete_pinecone_index(index_name=index_name)
            return None

        elif (
            index_name != "all"
            and index_name in pc.list_indexes().names()
            and delete == False
        ):
            print(f"Index {index_name} already exists. Loading embeddings ...", end="")
            vector_store = Pinecone.from_existing_index(index_name, embeddings)
            print("Ok")
        else:
            print(f"Creating index {index_name} and embeddings ...", end="")
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=PodSpec(environment="gcp-starter"),
            )
            vector_store = Pinecone.from_documents(
                chunks, embeddings, index_name=index_name
            )
            print("Ok")

        return vector_store

    insert_or_fetch_embeddings(
        index_name="langchain-gist-of-rag", chunks=texts, delete=False
    )

    # print("Ok")

    # print("Ingesting, Embedding, and Storing to Pinecone ... ")

    # PineconeVectorStore.from_documents(
    #     texts, embeddings, index_name=os.environ["INDEX_NAME"]
    # )

    print("Finished")
