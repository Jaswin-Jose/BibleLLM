import lightning as L
import gradio as gr
import torch
import faiss
import numpy as np
import json

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Load embedding model (UAE)
embed_model_name = "WhereIsAI/UAE-Large-V1"
embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
embed_model = AutoModel.from_pretrained(embed_model_name).eval()

# Load LLM (instruction-tuned, better than Falcon)
llm_name = "mistralai/Mistral-7B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_name).eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_model.to(device)
llm_model.to(device)

# Load data
chunks = json.load(open("chunks.json", "r", encoding="utf-8"))

index, embeddings = load_embeddings_and_build_index("embeddings.npy")

def load_embeddings_and_build_index(path="embeddings.npy"):
    # Load embeddings from .npy file
    embeddings = np.load(path).astype("float32")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index (cosine similarity via IndexFlatIP)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings

# Embedding function
def get_embedding(text):
    inputs = embed_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
        hidden_states = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        summed = (hidden_states * mask).sum(dim=1)
        count = mask.sum(dim=1)
        embedding = (summed / count).cpu().numpy()
    return embedding[0].astype("float32")

# RAG generation
def answer_query(query):
    query_emb = get_embedding(query).reshape(1, -1)
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb, k=5)
    top_chunks = [chunks[i]["content"] for i in I[0]]

    context = "\n\n".join(top_chunks)
    prompt = f"""You are a compassionate biblical assistant. Use the context to give a comforting, helpful response.

Context:
{context}

Question: {query}
Answer:"""

    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = llm_model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7
        )
    return llm_tokenizer.decode(output[0], skip_special_tokens=True)

# Gradio interface
ui = gr.Interface(fn=answer_query, inputs="text", outputs="text", title="Bible Q&A (RAG)")

# Lightning entrypoint
class BibleRAGApp(L.LightningApp):
    def configure_layout(self):
        return ui

app = BibleRAGApp()
if __name__ == "__main__":
    app.run()
