import torch
import torch.nn as nn
from torchvision import models, transforms
from getkeys import key_check
import numpy as np
from grabscreen import grab_screen
import cv2
import time
import keys as kk 
import os
from matplotlib import pyplot as plt
model_path = 'epoch-4-resnet18_model-9.pth'
k = kk.Keys()
class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        # Example: Use a pre-trained ResNet18 model
        self.resnet18 = models.resnet18(pretrained=True)
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)

model = CustomModel(5)
model.resnet18.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
stoped = False
paused = True
def move(c):
	if c == 0:
		k.directKey('w')
		k.directKey('d', k.key_release)
		k.directKey('a', k.key_release)
		k.directKey('s', k.key_release)
	elif c == 1:
		k.directKey('a')
		k.directKey('d', k.key_release)
		k.directKey('w', k.key_release)
		k.directKey('s', k.key_release)
	elif c == 2:
		k.directKey('s')
		k.directKey('d', k.key_release)
		k.directKey('w', k.key_release)
		k.directKey('a', k.key_release)
	elif c == 3:
		k.directKey('d')
		k.directKey('a', k.key_release)
		k.directKey('w', k.key_release)
		k.directKey('s', k.key_release)
	elif c == 4:
		pass

while not stoped:
	keys = key_check()
	if 'R' in keys:
		paused = not paused 
		print(f'Status: {str(paused)}')
		time.sleep(0.5)
	if 'Q' in keys:
		paused = not paused
		stoped = not stoped
		break
	if not paused:
		screen = grab_screen(region=(515,245,1400,1045))
		#screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
		screen = cv2.resize(screen, (75,75))
		screen = transform(screen)
		screen = screen.unsqueeze(0)
		with torch.no_grad():
			output = model(screen)
		predicted_class = torch.argmax(output).item()
		move(predicted_class)
		