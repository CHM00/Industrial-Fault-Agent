import os
import sys
import time
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, connections, utility
from dotenv import load_dotenv
from sop_documents import SOP_DOCUMENTS

load_dotenv()


COLLECTION_NAME = "industrial_fault_knowledge"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_DIM = 2560


def get_embedding(text: str) -> list:
    import requests
    api_key = os.environ.get("ARK_API_KEY")
    base_url = os.environ.get("ARK_BASE_URL", "https://api.siliconflow.cn/v1")
    url = f"{base_url}/embeddings"
    payload = {"model": EMBEDDING_MODEL, "input": text}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def connect_milvus():
    host = os.environ.get("MILVUS_HOST", "")
    port = os.environ.get("MILVUS_PORT", "19530")
    user = os.environ.get("MILVUS_USER", "root")
    password = os.environ.get("MILVUS_PASSWORD", "")
    uri = os.environ.get("URL", "")
    token = os.environ.get("Token", "")

    if host:
        connect_kwargs = {"alias": "fault_conn", "host": host, "port": port}
        if user:
            connect_kwargs["user"] = user
        if password:
            connect_kwargs["password"] = password
        connections.connect(**connect_kwargs)
        print(f"[Milvus] connected to {host}:{port}")
    elif uri:
        connect_kwargs = {"alias": "fault_conn", "uri": uri}
        if token:
            connect_kwargs["token"] = token
        connections.connect(**connect_kwargs)
        print("[Milvus] connected to Zilliz Cloud")
    else:
        from milvus_lite import MilvusLite
        local_path = str(os.path.abspath("output/milvus_local.db"))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        connections.connect(alias="fault_conn", uri=local_path)
        print(f"[Milvus] connected to local file: {local_path}")


def create_collection() -> Collection:
    if utility.has_collection(COLLECTION_NAME, using="fault_conn"):
        collection = Collection(name=COLLECTION_NAME, using="fault_conn")
        print(f"[Milvus] collection '{COLLECTION_NAME}' already exists, dropping and recreating...")
        collection.drop()

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    ]
    schema = CollectionSchema(fields, description="工业故障诊断SOP知识库")
    collection = Collection(name=COLLECTION_NAME, schema=schema, using="fault_conn")

    index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    collection.create_index(field_name="vector", index_params=index_params)
    collection.create_index(field_name="category", index_params={"index_type": "Trie"})
    print(f"[Milvus] collection '{COLLECTION_NAME}' created with dim={EMBEDDING_DIM}")
    return collection


def insert_documents(collection: Collection):
    print(f"[Embedding] starting embedding for {len(SOP_DOCUMENTS)} documents...")
    titles = []
    contents = []
    categories = []
    vectors = []

    for i, doc in enumerate(SOP_DOCUMENTS):
        embed_text = f"{doc['title']} {doc['category']} {doc['content']}"
        vec = get_embedding(embed_text)
        titles.append(doc["title"])
        contents.append(doc["content"])
        categories.append(doc["category"])
        vectors.append(vec)
        print(f"  [{i+1}/{len(SOP_DOCUMENTS)}] embedded: {doc['title']}")

    data = [vectors, titles, contents, categories]
    collection.insert(data)
    collection.flush()
    print(f"[Milvus] inserted {len(SOP_DOCUMENTS)} documents into '{COLLECTION_NAME}'")


def verify_search(collection: Collection):
    collection.load()
    test_query = "PLC通讯故障诊断步骤"
    query_vec = get_embedding(test_query)
    search_params = {"metric_type": "IP", "params": {"nlist": 128}}
    results = collection.search(
        data=[query_vec],
        anns_field="vector",
        param=search_params,
        limit=3,
        output_fields=["title", "category", "content"],
    )
    print(f"\n[Verify] search query: '{test_query}'")
    for hit in results[0]:
        print(f"  score={hit.distance:.4f} | title={hit.entity.get('title')} | category={hit.entity.get('category')}")
    print("[Verify] search verification passed!\n")


def main():
    connect_milvus()
    collection = create_collection()
    insert_documents(collection)
    verify_search(collection)
    print("[Done] knowledge base initialization complete!")


if __name__ == "__main__":
    main()