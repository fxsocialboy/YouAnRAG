import os
import re
from pathlib import Path
import jieba
import markdown2
import json

# 设置路径和参数
input_folder = './final_mds'  # 替换为你的 md 文件夹路径
chunk_max_chars = 250  # 每个 chunk 最大字符数

def clean_markdown(md_text):
    html = markdown2.markdown(md_text)
    text = re.sub(r'<[^>]+>', '', html)  # 移除 HTML 标签
    text = re.sub(r'\s+', '', text).strip()  # 移除多余空格
    return text

def split_sentences(text):
    pattern = re.compile(r'[^！？。；\n]*[！？。；]?')
    return [s for s in pattern.findall(text) if s.strip()]

def split_into_chunks(text, max_chars=500):
    sentences = split_sentences(text)
    chunks = []
    current_chunk = ''
    for sent in sentences:
        if len(current_chunk) + len(sent) <= max_chars:
            current_chunk += sent
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def process_md_files(folder_path):
    all_chunks = []
    for file in Path(folder_path).glob('*.md'):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                raw_md = f.read()
            clean_text = clean_markdown(raw_md)
            chunks = split_into_chunks(clean_text, max_chars=chunk_max_chars)
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "source_file": file.name,
                    "chunk_index": idx,
                    "content": chunk
                })
        except Exception as e:
            print(f"读取 {file.name} 时出错: {e}")
    return all_chunks

# 执行处理
chunks = process_md_files(input_folder)

print(len(chunks))

with open("md_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"已保存 {len(chunks)} 个 chunks 到 md_chunks.json")