import torch
model_path = "checkpoints/model_iter_0029.pt"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
print(checkpoint.values())