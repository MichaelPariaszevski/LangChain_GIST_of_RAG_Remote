def delete_pinecone_index(index_name="all"):
    import pinecone

    pc = pinecone.Pinecone()
    if index_name == "all":
        indexes = pc.list_indexes().names()
        print("Deleting all indexes ... ")
        for index in indexes:
            pc.delete_index(index)
        print("Deleted all Indexes")
    else:
        print(f"Deleting index {index_name} ... ", end="")
        pc.delete_index(index_name)
        print(f"Deleted index {index_name}")


def insert_or_fetch_embeddings(index_name: str, chunks, delete=False):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec

    pc = pinecone.Pinecone() # Pinecone api_key is added here
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

    print(pc.list_indexes())
    print("-" * 50)
    print(pc.list_indexes().names())
    print("-" * 50)

    if index_name == "all":
        delete_pinecone_index(index_name="all")
        print(pc.list_indexes())
        return None

    elif (
        index_name != "all"
        and index_name in pc.list_indexes().names()
        and delete == True
    ):
        delete_pinecone_index(index_name=index_name)
        print(pc.list_indexes())
        return None

    elif (
        index_name != "all"
        and index_name in pc.list_indexes().names()
        and delete != True
    ):
        print(f"Index {index_name} already exists. Loading embeddings ...", end="")
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print("Ok")
        return vector_store
    elif (
        index_name != "all"
        and index_name not in pc.list_indexes().names()
        and delete != True
    ):
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
    else: 
        print("Invalid set of parameters")

    print("-" * 200)
