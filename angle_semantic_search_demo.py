import os
import json
from typing import List, Tuple

import numpy as np
import streamlit as st
import faiss
from angle_emb import AnglE


# =============================
# 配置
# =============================
MODEL_NAME = "SeanLee97/angle-roberta-wwm-base-zhnli-v1"
POOLING_STRATEGY = "cls"
TOP_K = 5
CORPUS_PATH = "corpus.json"
INDEX_PATH = "faiss.index"
TEXTS_PATH = "texts.json"


# =============================
# 默认语料
# 你可以直接替换成自己的 FAQ / 文档标题 / 问答对
# =============================
DEFAULT_CORPUS = [
    "如何学习 Python 入门",
    "Python 编程基础教程",
    "如何配置 CUDA 和 PyTorch 环境",
    "深度学习模型训练显卡显存不足怎么办",
    "如何使用 FAISS 构建向量检索系统",
    "什么是语义搜索，与关键词搜索有什么区别",
    "AnglE 文本向量模型如何进行微调",
    "SimCSE 和 SBERT 有什么区别",
    "如何使用 HuggingFace 下载预训练模型",
    "机器学习和深度学习有什么区别",
    "如何进行文本相似度计算",
    "如何构建 FAQ 智能问答系统",
    "什么是向量数据库",
    "如何用 Streamlit 快速搭建演示系统",
    "RAG 中为什么需要文本 embedding",
]


# =============================
# 工具函数
# =============================
def ensure_corpus_exists() -> None:
    if not os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CORPUS, f, ensure_ascii=False, indent=2)


def load_corpus() -> List[str]:
    ensure_corpus_exists()
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("corpus.json 必须是字符串列表")
    return data


@st.cache_resource(show_spinner=True)
def load_model() -> AnglE:
    device_msg = "CUDA" if os.environ.get("CUDA_VISIBLE_DEVICES", None) is not None else "auto"
    st.info(f"正在加载 AnglE 模型：{MODEL_NAME}（device={device_msg}）")
    try:
        model = AnglE.from_pretrained(MODEL_NAME, pooling_strategy=POOLING_STRATEGY).cuda()
    except Exception:
        # 没有 GPU 时回退 CPU
        model = AnglE.from_pretrained(MODEL_NAME, pooling_strategy=POOLING_STRATEGY)
    return model


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


@st.cache_resource(show_spinner=True)
def build_or_load_index(_model: AnglE) -> Tuple[faiss.Index, List[str]]:
    texts = load_corpus()

    if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(TEXTS_PATH, "r", encoding="utf-8") as f:
            saved_texts = json.load(f)
        if saved_texts == texts:
            return index, texts

    embeddings = _model.encode(texts, to_numpy=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings = l2_normalize(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 归一化后，内积≈余弦相似度
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    return index, texts


def search(query: str, model: AnglE, index: faiss.Index, texts: List[str], top_k: int = TOP_K):
    q = model.encode([query], to_numpy=True)
    q = np.asarray(q, dtype=np.float32)
    q = l2_normalize(q)

    scores, indices = index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text": texts[idx],
            "score": float(score),
            "index": int(idx)
        })
    return results


# =============================
# 页面
# =============================
def main():
    st.set_page_config(page_title="AnglE 语义搜索 Demo", page_icon="🔎", layout="wide")
    st.title("🔎 基于 AnglE 的语义搜索 Demo")
    st.caption("输入一句自然语言查询，系统将使用 AnglE 生成文本向量，并通过 FAISS 检索最相似文本。")

    with st.sidebar:
        st.header("系统说明")
        st.write("- 文本编码：AnglE")
        st.write("- 向量检索：FAISS")
        st.write("- 相似度：归一化向量内积（近似余弦相似度）")
        st.write("- 语料文件：corpus.json")
        st.write("- 你可以直接编辑 corpus.json 替换成自己的 FAQ / 文档标题")
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=5)

        st.divider()
        st.subheader("默认测试查询")
        sample_queries = [
            "python 怎么入门",
            "为什么需要向量数据库",
            "怎么做文本相似度匹配",
            "如何训练 AnglE 模型",
            "RAG 为什么要做检索",
        ]
        for q in sample_queries:
            if st.button(q):
                st.session_state["query"] = q

    model = load_model()
    index, texts = build_or_load_index(model)

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("请输入查询语句", value=st.session_state.get("query", "python 怎么学习"))
    with col2:
        run = st.button("开始搜索", use_container_width=True)

    if run or query:
        results = search(query, model, index, texts, top_k=top_k)

        st.subheader("检索结果")
        for rank, item in enumerate(results, start=1):
            with st.container(border=True):
                st.markdown(f"**Top {rank}**")
                st.write(item["text"])
                st.write(f"相似度：`{item['score']:.4f}`")

    st.divider()
    st.subheader("当前语料")
    st.code(json.dumps(texts, ensure_ascii=False, indent=2), language="json")

    st.subheader("如何运行")
    st.code(
        "pip install streamlit faiss-cpu angle-emb\n"
        "streamlit run angle_semantic_search_demo.py",
        language="bash",
    )


if __name__ == "__main__":
    main()
