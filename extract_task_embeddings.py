import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

tasks = [
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

embeddings_tensor = model.encode(tasks, convert_to_tensor=True)   # shape (14,384)

# Save the original FP32 embeddings
torch.save(embeddings_tensor, "task_embeddings.pt")
print(f"\nSaved FP32 embeddings to task_embeddings.pt with shape {embeddings_tensor.shape}")

# Quantize to symmetric INT8 for FPGA BRAM
print("\nQuantizing embeddings for FPGA BRAM...")
scale = embeddings_tensor.abs().max() / 127.0
embeddings_int8 = (embeddings_tensor / scale).round().clamp(-128, 127).to(torch.int8)

# Save quantized version with scale factor
torch.save({'int8_data': embeddings_int8, 'scale': scale}, 'task_embeddings_int8.pt')
print("Saved INT8 embeddings to task_embeddings_int8.pt")