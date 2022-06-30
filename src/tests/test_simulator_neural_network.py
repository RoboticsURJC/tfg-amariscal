from jetcam.video_camera import VideoCamera
import torchvision.transforms as transforms
import torchvision
import torch
import PIL.Image
import cv2

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()


def preprocess(image):
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


CATEGORIES = ['apex']
device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 2 * len(CATEGORIES))
model = model.to(device)
model.load_state_dict(torch.load(
    '../models/road_following_model_sim.pth'))
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
        x = int(224 * (x / 2.0 + 0.5)) + 20
        y = int(224 * (y / 2.0 + 0.5)) + 50
        prediction = image.copy()
        prediction = cv2.circle(prediction, (x, y), 8, (255, 0, 0), 3)
        cv2.imshow('Image', prediction)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
