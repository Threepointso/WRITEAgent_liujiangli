import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# 本地模型路径（已下载到项目中）
LOCAL_MODEL_PATH = Path(__file__).parent / "models" / "BAAI" / "bge-base-zh-v1___5"

class PaperVectorStore:
    """
    持久化论文向量库
    功能：Reader 解析完的论文自动存入，下次写综述直接语义检索
    """
    def __init__(self, store_path="./paper_vector_store"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(exist_ok=True)
        self.encoder = SentenceTransformer(str(LOCAL_MODEL_PATH))
        self.papers = []
        self._load_or_create_index()

    def _load_or_create_index(self):
        """加载已有向量索引 / 或创建新的空索引"""
        index_file = self.store_path / "faiss.index"      # FAISS 向量索引文件
        meta_file = self.store_path / "papers.json"       # 论文元数据文件
        if index_file.exists() and meta_file.exists():
            # 已有向量库 → 加载索引和论文列表
            self.index = faiss.read_index(str(index_file))
            with open(meta_file, 'r', encoding='utf-8') as f:
                self.papers = json.load(f)
            print(f"   ✅ 加载已有向量库：{len(self.papers)} 篇论文")
        else:
            # 首次运行 → 创建空索引（768 维内积索引）
            self.index = faiss.IndexFlatIP(768)
            self.papers = []
            print("   📦 创建新的向量库")    

    def _save(self):
        """保存向量索引和论文元数据到磁盘"""
        faiss.write_index(self.index, str(self.store_path / "faiss.index"))
        with open(self.store_path / "papers.json", 'w', encoding='utf-8') as f:
            json.dump(self.papers, f, ensure_ascii=False, indent=2)
    def add_papers(self, new_papers: list):
        """新增论文（自动去重，已存在的跳过）"""
        existing_titles = {p["title"] for p in self.papers}          # 已有论文标题集合
        truly_new = [p for p in new_papers if p["title"] not in existing_titles]  # 过滤重复
        if not truly_new:
            print(f"   ⏭️ 所有 {len(new_papers)} 篇论文已存在，跳过")
            return
        texts = [p["title"] + " " + p.get("abstract", "")[:300] for p in truly_new]
        embeddings = self.encoder.encode(texts, normalize_embeddings=True)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.papers.extend(truly_new)
        self._save()
        print(f"   ✅ 向量库新增 {len(truly_new)} 篇（跳过 {len(new_papers) - len(truly_new)} 篇重复），共 {len(self.papers)} 篇")


    def search(self, query: str, k: int = 10) -> list:
        """语义检索：根据查询返回最相似的 k 篇论文"""
        q_vec = self.encoder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(q_vec, dtype=np.float32), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.papers) and score > 0.3:
                results.append({**self.papers[idx], "similarity": float(score)})
        return results

    def stats(self) -> dict:
        """返回向量库统计数据"""
        return {"total_papers": len(self.papers), "index_size": self.index.ntotal}


# 全局单例，所有模块共享
vector_store = PaperVectorStore()