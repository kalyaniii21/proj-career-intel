import pandas as pd
import os
import json

raw_csv_path = os.path.join("data", "job_descriptions.csv")  
sample_json_path = os.path.join("data", "career_intelligence_corpus.json")

print("🔄 Loading your massive Kaggle CSV file...")
chunks = [chunk for chunk in pd.read_csv(raw_csv_path, chunksize=10000)]
df_all = pd.concat(chunks, axis=0)

# ─── DYNAMIC HEADER DETECTION ───
title_col = next((col for col in df_all.columns if "title" in col.lower() or "role" in col.lower()), None)
desc_col = next((col for col in df_all.columns if "desc" in col.lower()), None)

if not title_col or not desc_col:
    raise KeyError(f"❌ Could not map column structures automatically. Found keys: {list(df_all.columns)}")

print(f"📊 Using columns -> Title: '{title_col}', Description: '{desc_col}'")

target_keywords = ["Backend", "Frontend", "DevOps", "Data Analyst", "Software", "Developer"]
filtered_rows = []
rows_per_category = 90  # Pulling 90 items per group to confidently build a solid 500+ ecosystem

for keyword in target_keywords:
    matched_df = df_all[df_all[title_col].str.contains(keyword, case=False, na=False)]
    sample_category = matched_df.head(rows_per_category)
    filtered_rows.append(sample_category)
    print(f"🎯 Extracted {len(sample_category)} pristine rows for: {keyword}")

df_final_sample = pd.concat(filtered_rows, axis=0).drop_duplicates(subset=[desc_col])

clean_corpus = []
for index, row in df_final_sample.iterrows():
    clean_corpus.append({
        "entity_id": f"kaggle_node_{1000 + index}",
        "title": row[title_col],
        "type": "Role",
        "description": row[desc_col],
        "source_metadata": {
            "authority_type": "official_corporate_framework",
            "base_trust_score": 0.85
        }
    })

with open(sample_json_path, "w") as f:
    json.dump(clean_corpus, f, indent=5)

print(f"\n🚀 Success! Compiled {len(clean_corpus)} targeted records into your active database file!")