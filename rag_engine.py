import os
import chromadb
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class CareerRAGEngine:
    def __init__(self):
        
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.chroma_collection = self.chroma_client.get_collection("career_intelligence_vectors")
        
        
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        
        if "+s://" in uri:
            uri = uri.replace("+s://", "+ssc://")
        elif "bolt+s://" in uri:
            uri = uri.replace("bolt+s://", "bolt+ssc://")
            
        self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.neo4j_driver.close()

    def retrieve_hybrid_context(self, search_query):
        print(f"🔍 Analyzing search query: '{search_query}'")
        
        
        vector_results = self.chroma_collection.query(
            query_texts=[search_query],
            n_results=3
        )
        
        documents = vector_results.get("documents", [[]])[0]
        metadatas = vector_results.get("metadatas", [[]])[0]
        
        vector_context = []
        retrieved_titles = []
        for doc, meta in zip(documents, metadatas):
            vector_context.append(f"Role: {meta['title']}\nDescription: {doc}")
            retrieved_titles.append(meta['title'])
            
        print(f"✅ ChromaDB pulled {len(vector_context)} semantic matches.")

        
        graph_context = []
        if retrieved_titles:
            with self.neo4j_driver.session() as session:
                # Find roles that are explicitly connected to our vector-matched roles
                query = """
                MATCH (a:Entity)-[:RELATED_TO]-(b:Entity)
                WHERE a.title IN $titles
                RETURN DISTINCT b.title as connected_title, b.description as connected_desc
                LIMIT 5
                """
                results = session.run(query, titles=retrieved_titles)
                for record in results:
                    graph_context.append(f"Connected Pathway Role: {record['connected_title']}\nContext: {record['connected_desc']}")
                    
            print(f"✅ Neo4j pulled {len(graph_context)} network relationship links.")

        
        full_context = "\n\n=== SEMANTIC RELEVANT ROLES ===\n" + "\n---\n".join(vector_context)
        if graph_context:
            full_context += "\n\n=== CONNECTED CAREER PATHWAY TRAJECTORIES ===\n" + "\n---\n".join(graph_context)
            
        return full_context

if __name__ == "__main__":
    # Test script standalone to verify it compiles and runs perfectly
    engine = CareerRAGEngine()
    try:
        # Testing with a classic query term
        test_context = engine.retrieve_hybrid_context("Backend Developer")
        print("\n🚀 --- SAMPLE RETRIEVED HYBRID CONTEXT FOR LLM ---")
        print(test_context[:1000] + "...\n[Truncated for logs]")
    finally:
        engine.close()