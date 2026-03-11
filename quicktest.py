from angle_emb import AnglE
from angle_emb.utils import cosine_similarity

model = AnglE.from_pretrained(
    'SeanLee97/angle-roberta-wwm-base-zhnli-v1',
    pooling_strategy='cls'
).cuda()

texts = [
    "我爱吃菠萝",
    "我爱吃凤梨",
    "我爱吃苹果"
]

vecs = model.encode(texts, to_numpy=True)

print(cosine_similarity(vecs[0], vecs[1]))
print(cosine_similarity(vecs[0], vecs[2]))