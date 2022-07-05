from Pokemon import PokemonAI, PokemonType
from torchvision import models
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
import CAMHelpers as cam

classes = [t for t in [PokemonType(i) for i in range(18)] if t not in [PokemonType.Flying]]

model = models.resnet50()
model.fc = nn.Linear(2048, len(classes))
trained_weights = torch.load("trained_model")
model.load_state_dict(trained_weights)
model.eval()

files = []
for f in files:

    img = Image.open(f).convert("RGB")
    prediction, confidence, class_probs, tensor = PokemonAI.predict_img(img_path, model, classes)

    target_layer = model.layer4[-1]
    cam_generator = GradCAM(model=model, target_layers=[target_layer])

    cam_output = cam_generator(input_tensor=tensor.unsqueeze(0))
    cam_output_img = cam.createImage(cam_output,"jet")
    overlayed = cam.overlayCAM(img,cam_output_img, 0.5)

    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.title(f"{prediction} ({confidence:.2f}%)")
    plt.subplot(2,2,2)
    probs = [prob[1] for prob in class_probs]
    plt.barh([str(c) for c in classes],probs)
    plt.xlim([0,100])
    plt.title("Probability Outputs")
    plt.subplot(2,2,3)
    plt.imshow(overlayed)
    plt.title("Areas that most contributed for the result")

    plt.show()


