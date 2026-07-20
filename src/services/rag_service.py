from embedding.embedder import Embedder
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.reranker import Reranker
from generation.prompt_builder import PromptBuilder
from generation.llm import LLM

class RAGService:
    def __init__(self):
        self.embedder = Embedder()
        self.retriever = FAISSRetriever()
        self.reranker = Reranker()
        self.prompt_builder = PromptBuilder()
        self.llm = LLM()

    def ask(self, query: str):
        # 1. Embed Query
        query_embedding = self.embedder.encode([query])
        
        # 2. Retrieve
        retrieval_results = self.retriever.retrieve(query_embedding, k=20)
        
        # 3. Rerank
        reranked_results = self.reranker.rerank(query, retrieval_results)
        
        # 4. Take Top-3 to reduce context length
        top_k_results = reranked_results[:3]
        
        # 5. Extract Documents
        documents = [res.document for res in top_k_results]
        
        # 6. Build Prompt
        prompt = self.prompt_builder.build(query, documents)
        
        # 7. Generate Answer Stream
        answer_stream = self.llm.generate_stream(prompt)
        
        return {
            "question": query,
            "answer_stream": answer_stream,
            "sources": top_k_results
        }
