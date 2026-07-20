import sys
from pathlib import Path

# Add src to sys.path so modules can be imported
sys.path.append(str(Path(__file__).resolve().parent))

from services.rag_service import RAGService

def main():
    print("Initializing RAG Service...")
    rag_service = RAGService()
    
    print("=" * 60)
    print("RAG System Ready. Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    
    while True:
        try:
            query = input("\nQuestion: ").strip()
            if query.lower() in ['quit', 'exit']:
                break
                
            if not query:
                continue
                
            print("\nGenerating answer...")
            result = rag_service.ask(query)
            
            print("\n" + "=" * 60)
            print("ANSWER")
            print("=" * 60)
            
            for token in result["answer_stream"]:
                print(token, end="", flush=True)
            print() # Print newline when done
            
            print("\n" + "=" * 60)
            print("SOURCES")
            print("=" * 60)
            for i, res in enumerate(result["sources"], 1):
                print(f"[{i}] Score: {res.score:.4f} | {res.document.title}")
            print("=" * 60)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
