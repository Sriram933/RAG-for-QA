# RAG-Based Question Answering System

This project implements a Retrieval-Augmented Generation (RAG) pipeline for answering queries using Wikipedia snippets.
It retrieves relevant passages from a FAISS-based vector index and generates responses using a T5-based transformer model.
## Features
- Uses FAISS for efficient similarity search on Wikipedia passages.
- Sentence-Transformer for encoding text into dense embeddings.
- Flan-T5 for natural language generation of answers.
- Implements a full RAG pipeline: retrieval + generation.

## Installation

Ensure you have the required dependencies installed:

```python
pip install datasets sentence-transformers faiss-cpu transformers torch
```
## Usage
```python
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load_corpus():
    dataset = load_dataset("wiki_snippets", "wiki40b_en_100_0", split="train[:10000]")
    corpus = [doc['passage_text'] for doc in dataset]
    return corpus
# Load and preprocess corpus
corpus = load_corpus()
unique_corpus = list(set(corpus))  # Remove duplicates
print(f"Total passages: {len(corpus)}, Unique passages: {len(unique_corpus)}")

# Load the sentence transformer model
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
corpus_embeddings = model.encode(unique_corpus, convert_to_tensor=True)

# Build FAISS index
embedding_dim = corpus_embeddings.shape[1]
res = faiss.StandardGpuResources()
index = faiss.IndexFlatL2(embedding_dim)
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
gpu_index.add(corpus_embeddings.cpu().detach().numpy())

def retrieve(query, top_k=15):
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().detach().numpy()
    distances, indices = gpu_index.search(query_embedding, top_k)
    return [unique_corpus[i] for i in indices[0]]

# Load T5 generator model
generator = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

def generate_answer(context, query):
    input_text = f"query: {query} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = generator.generate(inputs.input_ids, max_length=150, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def rag_pipeline(query, top_k=15):
    retrieved_docs = retrieve(query, top_k)
    context = " ".join(retrieved_docs)
    return generate_answer(context, query)

# Example Queries
query = "What are the contributions of Ági Szalóki to music?"
answer = rag_pipeline(query)
print("Query:", query)
print("Answer:", answer)

query = "Who acquired the newspaper La Esfera?"
answer = rag_pipeline(query)
print("Query:", query)
print("Answer:", answer)
```

## Explanation

### 1.Retrieval:

- Wikipedia snippets are encoded into dense embeddings.
- FAISS is used for fast nearest-neighbor search.

### 2.Generation:
- The retrieved text is fed into Flan-T5 to generate answers.

## Future Improvements

- Support for larger Wikipedia dataset.
- Experiment with different embedding models.
- Implement a web-based interface.










