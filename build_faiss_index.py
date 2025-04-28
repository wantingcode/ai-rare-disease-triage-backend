import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load symptom-disease pairs
with open('symptom_disease_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Load model
model = SentenceTransformer('BAAI/bge-small-zh')

# Encode symptoms
symptoms = [item['symptom'] for item in dataset]
embeddings = model.encode(symptoms, normalize_embeddings=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype=np.float32))

# Save index and mapping
faiss.write_index(index, 'symptom_index.faiss')
with open('symptom_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print("FAISS index and symptom mapping saved.")