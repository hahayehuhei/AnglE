import sys
import io
import torch
from angle_emb import AnglE
from torch import nn

# --- 1. 解决 Windows 终端乱码 ---
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- 2. 检查并准备设备 (GPU 或 CPU) ---
# 这一步会自动判断你的电脑有没有显卡驱动
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")

# --- 3. 加载模型并移动到 GPU ---
print("正在加载模型...")
# 注意：在这里加了 .cuda()。如果你写了 device，也可以用 .to(device)
model = AnglE.from_pretrained('SeanLee97/angle-roberta-wwm-base-zhnli-v1', pooling_strategy='cls').to(device)

# --- 4. 准备知识库 ---
knowledge_base = [
    "如何使用深度学习进行图像识别？",
    "北京明天的天气预报是晴天。",
    "AoE模型通过复数空间优化，比普通的向量模型更聪明。",
    "如何制作一道好吃的红烧肉？",
    "今天心情不错，天气很好。"
]

print("正在将知识库向量化 (使用 GPU)...")
# 生成向量
kb_vectors = model.encode(knowledge_base, to_numpy=True)
# 将向量转为 Torch Tensor 并存放到 GPU 上，方便快速对比
kb_vectors_torch = torch.from_numpy(kb_vectors).to(device)

def search(query_text):
    # 编码用户提问，并转到 GPU
    query_vector = model.encode(query_text, to_numpy=True)
    query_vector_torch = torch.from_numpy(query_vector).to(device)
    
    # 在 GPU 上计算余弦相似度
    cos = nn.CosineSimilarity(dim=1)
    scores = cos(query_vector_torch, kb_vectors_torch)
    
    # 找到最高分的索引
    best_idx = torch.argmax(scores).item()
    return knowledge_base[best_idx], scores[best_idx].item()

# --- 5. 交互界面 ---
print("\n" + "="*30)
print("系统已就绪！(GPU已开启)")
print("请在下方 'Search >>>' 处输入并回车。")
print("="*30)

while True:
    try:
        # 确保在“终端”里运行，这里才能输入
        user_input = input("Search >>> ")
        if user_input.lower() == 'q':
            break
        
        if not user_input.strip():
            continue

        result, score = search(user_input)
        print(f"匹配结果: {result}")
        print(f"相似度得分: {score:.4f}")
    except Exception as e:
        print(f"运行出错: {e}")