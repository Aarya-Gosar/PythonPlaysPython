from getkeys import key_check
import numpy as np
from grabscreen import grab_screen
import cv2
import time
import keys as k 
import os
from matplotlib import pyplot as plt
#import torch
def keys_to_output(keys):
	#[A,W,D]

	output = [0,0,0,0,0]

	if 'W' in keys:
		output[0] = 1
	elif 'A' in keys:
		output[1] = 1
	elif 'S' in keys:
		output[2] = 1
	elif 'D' in keys:
		output[3] = 1
	else:
		output[4] = 1

	return output
file_name = "test_data-9-imgs.npy"
file_name2 ="test_data-9-keys.npy"
	
if os.path.isfile(file_name):
	print('File exists!')
	training_data = list(np.load(file_name , allow_pickle=True))
	training_labels = list(np.load(file_name2 , allow_pickle=True))
else:
	print("creating a new file")
	training_data = []
	training_labels = []

stoped = False
paused = True
while not stoped:
	keys = key_check()
	if not paused:
		
		screen = grab_screen(region=(515,245,1400,1045))
		#screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
		screen = cv2.resize(screen, (75,75))
		#plt.imshow(screen , cmap = "gray")
		#plt.show()
		output = keys_to_output(keys)
		training_data.append(screen)
		training_labels.append(output)
		if len(training_data) % 50 == 0:
			print(len(training_data))
			np.save(file_name, np.array(training_data))
			np.save(file_name2,np.array(training_labels))
			
	if 'R' in keys:
		paused = not paused 
		print(f'Status: {str(paused)}')
		time.sleep(0.5)
	if 'Q' in keys:
		paused = not paused
		stoped = not stoped
		break


