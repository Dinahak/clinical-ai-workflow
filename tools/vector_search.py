# tools/vector_search.py

import chromadb
from chromadb.utils import embedding_functions
from utils.models import PatientCase, AuditEntry


class VectorSearch:
    """
    Manages the ChromaDB vector store.
    Indexes clinical documents and retrieves
    the most relevant ones for a given patient case.
    """

    def __init__(self, persist_dir: str = "./embeddings/chroma_store"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="fm_guidelines",
            embedding_function=self.embed_fn
        )

    def index_documents(self, documents: list[dict]):
        """
        Add clinical documents to the vector store.
        Each document dict must have: id, text, metadata
        """
        existing = self.collection.get()["ids"]

        new_docs = [d for d in documents if d["id"] not in existing]
        if not new_docs:
            print("All documents already indexed.")
            return

        self.collection.add(
            ids=[d["id"] for d in new_docs],
            documents=[d["text"] for d in new_docs],
            metadatas=[d.get("metadata", {}) for d in new_docs]
        )
        print(f"Indexed {len(new_docs)} new documents.")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search the vector store with a query string.
        Returns top_k most relevant documents.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )

        if not results["documents"][0]:
            return []

        retrieved = []
        for i, doc in enumerate(results["documents"][0]):
            retrieved.append({
                "text":     doc,
                "metadata": results["metadatas"][0][i],
                "distance": round(results["distances"][0][i], 4)
            })
        return retrieved

    def count(self) -> int:
        return self.collection.count()