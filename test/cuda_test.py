import torch 

print("🔥 CUDA available:", torch.cuda.is_available())
print("🔥 Current GPU memory allocated:", torch.cuda.memory_allocated() // 1024**2, "MB")
print("🔥 Current GPU memory reserved:", torch.cuda.memory_reserved() // 1024**2, "MB")

