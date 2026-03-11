from angle_emb import AnglE
from angle_emb.utils import cosine_similarity

# 加载官方预训练 embedding 模型
angle = AnglE.from_pretrained(
    'WhereIsAI/UAE-Large-V1',
    pooling_strategy='cls'
).cuda()

texts = [
    'The weather is great!',
    'The weather is very good!',
    'I am going to bed.'
]

vecs = angle.encode(texts, to_numpy=True)

print("sim(0,1) =", cosine_similarity(vecs[0], vecs[1]))
print("sim(0,2) =", cosine_similarity(vecs[0], vecs[2]))
print("sim(1,2) =", cosine_similarity(vecs[1], vecs[2]))