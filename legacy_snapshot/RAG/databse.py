import json
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# === 参数设置 ===
model_path = "./bge-large-zh-v1.5"
instruction = "为这个句子生成表示(向量嵌入)以用于检索相关文章："
chunk_file = "./md_chunks.json"
faiss_index_file = "faiss_index.index"
metadata_file = "chunk_metadata.json"

# === 加载模型 ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# === 读取 chunks ===
with open(chunk_file, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [instruction + chunk["content"] for chunk in chunks]

# === 生成 embedding ===
def encode(text_list, batch_size=32):
    embeddings = []
    a = 0
    for i in tqdm(range(0, len(text_list), batch_size), desc="Embedding"):
        batch_texts = text_list[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
        a=a+1
        print(a)
        with torch.no_grad():
            output = model(**inputs)
            cls_embeddings = output.last_hidden_state[:, 0]  # [B, H]
            cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
            embeddings.append(cls_embeddings.cpu())
    return torch.cat(embeddings, dim=0).numpy()  # 返回为 numpy 数组

embeddings = encode(texts)  # shape: [N, 1024]

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)  # 使用 L2 距离，可选：IndexFlatIP（内积）
index.add(embeddings)  # 添加向量

# === 保存索引 & 元数据 ===
faiss.write_index(index, faiss_index_file)
with open(metadata_file, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"已保存向量索引到 {faiss_index_file}，共 {len(chunks)} 条")