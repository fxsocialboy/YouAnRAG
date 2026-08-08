import torch
from transformers import AutoTokenizer, AutoModel

# 模型路径（可替换为本地目录）
model_path = "./bge-large-zh-v1.5"

# 加载模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# 句子 + 官方建议的 instruction prompt
sentence = "如何提高睡眠质量？"
instruction = "为这个句子生成表示(向量嵌入)以用于检索相关文章："
text = instruction + sentence

# 编码输入，送入 GPU
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)

# 推理，提取 [CLS] 向量作为 embedding
with torch.no_grad():
    output = model(**inputs)
    cls_embedding = output.last_hidden_state[:, 0]  # [batch, hidden_dim]
    embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)  # L2 归一化

# 输出结果
print("句子 embedding 向量：", embedding[0][:10])  # 只展示前10维
print("向量维度：", embedding.shape)  # 应该是 [1, 1024]