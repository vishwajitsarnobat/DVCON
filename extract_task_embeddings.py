from sentence_transformers import SentenceTransformer
import torch

model=SentenceTransformer('all-MiniLM-L6-v2')

tasks=[
    "step on something",
    "sit comfortably",
    "place flowers",
    "get potatoes out of fire",
    "water plant",
    "get lemon out of tea",
    "dig hole",
    "open bottle of beer",
    "open parcel",
    "serve wine",
    "pour sugar",
    "smear butter",
    "extinguish fire",
    "pound carpet"
]

# 14x384 tensor
embeddings=model.encode(tasks,convert_to_tensor=True)
torch.save(embeddings,'task_embeddings.pt')