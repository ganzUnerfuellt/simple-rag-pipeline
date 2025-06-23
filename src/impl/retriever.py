from interface.base_datastore import BaseDatastore
from interface.base_retriever import BaseRetriever
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

class Retriever(BaseRetriever):
    def __init__(self, datastore: BaseDatastore):
        self.datastore = datastore

    def search(self, query: str, top_k: int = 10) -> list[str]:
        search_results = self.datastore.search(query, top_k=top_k * 3)
        if not search_results:
            print(f"Warning: No documents found for query: {query}")
            return [] # Avoid calling rerank with empty list
        reranked_results = self._rerank(query, search_results, top_k=top_k)
        return reranked_results

    def _rerank(
        self, query: str, search_results: list[str], top_k: int = 10
    ) -> list[str]:

        cohere_api_key = os.getenv("CO_API_KEY")
        if not cohere_api_key:
            print("cohere API key not found")
            
        co = cohere.ClientV2(api_key=cohere_api_key)
        response = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=search_results,
            top_n=top_k,
        )

        result_indices = [result.index for result in response.results]
        print(f"✅ Reranked Indices: {result_indices}")
        return [search_results[i] for i in result_indices]
