import faiss
import json
import torch
from transformers import AutoTokenizer, AutoModel

# === 配置 ===
model_path = "./bge-large-zh-v1.5"
faiss_index_file = "faiss_index.index"
metadata_file = "chunk_metadata.json"
instruction = ""

# === 加载模型 ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(
    model_path,
    device_map="cpu",
    dtype=torch.float32
)
model.eval()

# === 加载 FAISS 索引和 chunk 元信息 ===
index = faiss.read_index(faiss_index_file)
with open(metadata_file, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# === 检索函数 ===
def search(query, top_k=10):
    text = query
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
    with torch.no_grad():
        output = model(**inputs)
        cls_embedding = output.last_hidden_state[:, 0]
        embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1).cpu().numpy()
    D, I = index.search(embedding, top_k)
    return I[0], D[0]

# === 示例查询 ===
if __name__ == "__main__":
    query = input("请输入查询语句：")
    ids, scores = search(query, top_k=10)

    print(f"\n🔍 查询结果 Top 10：\n")
    for i, idx in enumerate(ids):
        chunk = chunks[idx]
        print(f"Top {i+1} | 相似度（L2距离）: {scores[i]:.4f}")
        print(f"文件: {chunk['source_file']} | Chunk #{chunk['chunk_index']}")
        print(f"内容: {chunk['content'][:100]}{'...' if len(chunk['content']) > 100 else ''}")
        print("-" * 80)
