import sys
from pathlib import Path

# Add src to sys.path so modules can be imported
sys.path.append(str(Path(__file__).resolve().parent))

from services.rag_service import RAGService

def test():
    print("Initializing RAG Service...")
    rag_service = RAGService()
    
    query = "How to sort a dictionary in Python?"
    print(f"Testing with query: '{query}'")
    
    print("\n1. Testing Embedder, Retriever, and Reranker...")
    try:
        query_embedding = rag_service.embedder.encode([query])
        print("Embedder: Success")
        
        retrieval_results = rag_service.retriever.retrieve(query_embedding, k=5)
        print(f"Retriever: Success. Got {len(retrieval_results)} results.")
        
        reranked_results = rag_service.reranker.rerank(query, retrieval_results)
        print(f"Reranker: Success. Top score: {reranked_results[0].score:.4f}")
        
        documents = [res.document for res in reranked_results]
        prompt = rag_service.prompt_builder.build(query, documents)
        print(f"PromptBuilder: Success. Prompt length: {len(prompt)} chars.")
        
    except Exception as e:
        print(f"FAILED before LLM step: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n2. Testing LLM Generation (Requires Ollama to be running)...")
    try:
        answer_stream = rag_service.llm.generate_stream(prompt)
        print("LLM: Success! Streaming response:\n")
        print("-" * 40)
        for token in answer_stream:
            print(token, end="", flush=True)
        print("\n" + "-" * 40)
    except Exception as e:
        print(f"LLM Generation FAILED (Is Ollama running?): {e}")

if __name__ == "__main__":
    test()
