import torch

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

data=torch.load('task_embeddings.pt')
print(type(data))

for i in range(14):
    print("Input sentence:",tasks[i])
    print("Embedding:",data[i])