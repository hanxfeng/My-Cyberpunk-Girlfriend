import json
import faiss
from sentence_transformers import SentenceTransformer

with open("templates/train.json", "r", encoding="utf-8") as f:
    knowledge_data = json.load(f)
embedding_model = SentenceTransformer("models/m3e-base")

documents = []
for item in knowledge_data:
    instruction = item.get("instruction", "")
    output = item.get("output", "")
    doc = f"问题：{instruction}\n回答：{output}"
    documents.append(doc)

doc_embeddings = embedding_model.encode(documents, show_progress_bar=True)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

faiss.write_index(index, "templates/index.faiss")