import faiss
import json
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
class RAG:
    def __init__(self, model_path, faiss_index_file, metadata_file, instruction=None):
        self.model_path = model_path
        self.faiss_index_file = faiss_index_file
        self.metadata_file = metadata_file
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model.eval()

        # === 加载 FAISS 索引和 chunk 元信息 ===
        self.index = faiss.read_index(faiss_index_file)
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    # === 检索函数 ===
    def search(self, query, top_k=10):
        text = query
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        with torch.no_grad():
            output = self.model(**inputs)
            cls_embedding = output.last_hidden_state[:, 0]
            embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1).cpu().numpy()
        D, I = self.index.search(embedding, top_k)
        return I[0], D[0]

    def invoke(self, query: str, top_k: int=10)->List[str]:
        final_docs = []
        ids, scores = self.search(query, top_k=top_k)
        for i, idx in enumerate(ids):
            chunk = self.chunks[idx]
            final_docs.append(chunk['content'])
        return final_docs

