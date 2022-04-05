# Modified from https://github.com/NVIDIA-AI-IOT/jetracer

from torch2trt import torch2trt
import torch
import torchvision

CATEGORIES = ['apex']
device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2 * len(CATEGORIES))
model = model.cuda().eval().half()

model.load_state_dict(torch.load('road_following_model.pth'))

data = torch.zeros((1, 3, 224, 224)).cuda().half()

model_trt = torch2trt(model, [data], fp16_mode=True)

torch.save(model_trt.state_dict(), 'road_following_model_trt.pth')
