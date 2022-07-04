from Pokemon import PokemonAI, PokemonType
from torchvision import models
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
import CAMHelpers as cam
import os
from tqdm import tqdm

classes = [t for t in [PokemonType(i) for i in range(18)] if t not in [PokemonType.Flying, PokemonType.Fairy, PokemonType.Ice]]

model = models.resnet50()
model.fc = nn.Linear(2048, len(classes))
trained_weights = torch.load("training1 - all but flying and fairy and ice")
model.load_state_dict(trained_weights)
model.eval()

images = os.listdir("anime_screenshots")

progress = tqdm(images)
for file in progress:
    progress.set_description(file)
    img_path = os.path.join('anime_screenshots', file)
    img = Image.open(img_path).convert("RGB")
    prediction, tensor = PokemonAI.predict_img(img_path, model, classes)

    target_layer = model.layer4[-1]
    cam_generator = GradCAM(model=model, target_layers=[target_layer])

    cam_output = cam_generator(input_tensor=tensor.unsqueeze(0))
    cam_output_img = cam.createImage(cam_output,"jet")
    overlayed = cam.overlayCAM(img,cam_output_img, 0.5)

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title(prediction)
    plt.subplot(1,2,2)
    plt.imshow(overlayed)
    plt.title("Areas that most contributed for the result")

    plt.show()


