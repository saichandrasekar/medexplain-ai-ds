import sys
sys.path.append(".")

# Must load model before any embedding calls
from backend.services.embedding_service import _load_model
_load_model()
print("Embedding model loaded")

from backend.services.ocr_service import extract_chunks
from backend.services.rag_service import ingest_documents, query, list_documents

# Step 1 — Ingest
pdf_path = "rag_docs/sample_medical_docs/cholesterol-in-adults.pdf"
chunks = extract_chunks(pdf_path)
print(f"Extracted {len(chunks)} chunks")

# Step 2 — Store
count = ingest_documents(chunks)
print(f"Stored {count} chunks in ChromaDB")

# Step 3 — List
docs = list_documents()
print(f"Documents in store: {docs}")

# Step 4 — Query
results = query("what is LDL cholesterol", top_k=3)
for i, r in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Source : {r['source']} (page {r['page']})")
    print(f"Score  : {r['relevance_score']}")
    print(f"Text   : {r['text'][:200]}")