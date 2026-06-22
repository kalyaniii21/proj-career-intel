import json
import os

def transform_kaggle_to_platform_schema():
    print("🔄 Parsing and transforming Kaggle sample data...")
    
    # Define paths
    input_path = os.path.join("data", "kaggle_sample.json")
    output_path = os.path.join("data", "career_intelligence_corpus.json")
    
    if not os.path.exists(input_path):
        print(f"❌ Error: Could not find {input_path}. Please make sure your sample is saved there.")
        return

    with open(input_path, "r") as f:
        kaggle_data = json.load(f)
        
    transformed_nodes = []
    
    for idx, job in enumerate(kaggle_data):
        # 1. Standardize the textual entity description
        # We combine title, responsibilities, and skills into a rich context block for ChromaDB
        combined_description = (
            f"Role: {job.get('Role', 'N/A')} (Job Title: {job.get('Job Title', 'N/A')}). "
            f"Company: {job.get('Company', 'N/A')}. "
            f"Core Description: {job.get('Job Description', 'N/A')} "
            f"Key Requirements and Skills: {job.get('skills', 'N/A')}. "
            f"Main Responsibilities: {job.get('Responsibilities', 'N/A')}."
        )
        
        # 2. Build the relationship array dynamically for the future Graph layer
        relationships = []
        if job.get("Company"):
            relationships.append({"target": job["Company"], "type": "HIRED_BY"})
            
        # 3. Format the metadata tracking fields for the Source Trust Matrix
        node = {
            "entity_id": f"kaggle_node_{1000 + idx}",
            "title": job.get("Role", "Unknown Role"),
            "type": "Role",
            "description": combined_description,
            "relationships": relationships,
            "source_metadata": {
                "authority_type": "official_corporate_framework", # Standardized fallback
                "base_trust_score": 0.85 # Baselines for public production data datasets
            }
        }
        
        transformed_nodes.append(node)
        
    # 4. Write back or append to your master platform corpus file
    with open(output_path, "w") as f:
        json.load # Clears/creates file fresh
        json.dump(transformed_nodes, f, indent=2)
        
    print(f"✅ Transformation complete! Appended {len(transformed_nodes)} records into {output_path}")

if __name__ == "__main__":
    transform_kaggle_to_platform_schema()