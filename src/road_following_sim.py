from jetcam.video_camera import VideoCamera
from utils import preprocess
import torchvision
import torch
import cv2

CATEGORIES = ['apex']
device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 2 * len(CATEGORIES))
model = model.to(device)
model.load_state_dict(torch.load(
    'road_following_model_sim_3_trained.pth'))
model = model.cuda().eval().half()

camera = VideoCamera(width=224, height=224)

STEERING_GAIN = 0.75
STEERING_BIAS = 0.00

while True:
    image = camera.read()
    if image is not None:
        image_preprocess = preprocess(image).half()
        output = model(image_preprocess).detach().cpu().numpy().flatten()
        x = float(output[0])
        y = float(output[0])
        # print("Output" + str(output))
        x = int(224 * (x / 2.0 + 0.5)) + 20
        y = int(224 * (y / 2.0 + 0.5)) + 50
        prediction = image.copy()
        prediction = cv2.circle(prediction, (x, y), 8, (255, 0, 0), 3)
        cv2.imshow('A', prediction)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
