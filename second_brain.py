import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def main():
    # Define the path to your CSV file in the Downloads directory
    downloads_csv = os.path.expanduser("~/Downloads/data.csv")
    if not os.path.exists(downloads_csv):
        print(f"Error: {downloads_csv} not found. Please ensure your CSV file is in the Downloads folder and named 'data.csv'.")
        return

    # Load CSV data
    df = pd.read_csv(downloads_csv)
    
    # Check if a "Text" column exists and has non-empty content
    if "Text" in df.columns and not df["Text"].fillna("").str.strip().eq("").all():
        texts = df["Text"].astype(str).tolist()
        print("Using the 'Text' column for embeddings.")
    else:
        # If "Text" is empty or missing, try combining other columns
        cols_to_combine = []
        for col in ["Ecosystem", "Data Sources/Tools", "Purpose", "Category"]:
            if col in df.columns:
                cols_to_combine.append(col)
        if cols_to_combine:
            df["CombinedText"] = df[cols_to_combine].fillna("").agg(" ".join, axis=1)
            texts = df["CombinedText"].tolist()
            print("The 'Text' column is empty or missing. Using combined text from columns:", cols_to_combine)
        else:
            print("Error: No 'Text' column found and no other columns available to combine.")
            return

    # Load the embedding model from Hugging Face (using all-MiniLM-L6-v2 as a starter)
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Compute embeddings for all texts
    print("Computing embeddings for your data...")
    embeddings = model.encode(texts, convert_to_tensor=True)

    # Prompt user for a query
    query = input("Enter your query: ")

    # Compute the embedding for the query
    query_embedding = model.encode([query], convert_to_tensor=True)
    # Calculate cosine similarity between the query and all document embeddings
    cos_sim = cosine_similarity(query_embedding.cpu().numpy(), embeddings.cpu().numpy())[0]

    # Get indices of the top 5 most similar rows
    top_k = 5
    top_k_idx = np.argsort(cos_sim)[::-1][:top_k]
    results = df.iloc[top_k_idx]

    print("\nTop matching rows:")
    print(results)
    print("\nSimilarity scores:")
    print(cos_sim[top_k_idx])

if __name__ == "__main__":
    main()

