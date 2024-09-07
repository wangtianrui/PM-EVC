import torch

checkpoint = torch.load(r"/Work20/2023/wangtianrui/model_temp/whisper/medium.pt", map_location="cpu")

print(checkpoint["dims"])

checkpoint = torch.load(r"/Work20/2023/wangtianrui/model_temp/whisper/small.pt", map_location="cpu")

print(checkpoint["dims"])