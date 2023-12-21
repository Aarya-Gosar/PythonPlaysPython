import numpy as np 
import random
from matplotlib import pyplot as plt
file_name = "test_data-9-imgs.npy"
file_name2 = "test_data-9-keys.npy"
data = np.load(file_name , allow_pickle=True)
lable = np.load(file_name2,allow_pickle=True)


lefts = []
rights = []
forwards = []
ups = []
downs = []


for i in range(len(lable)):
	#print(i)
	if lable[i][0] == 1:
		ups.append([data[i],lable[i]])
		

	elif lable[i][1] == 1:		
		lefts.append([data[i],lable[i]])
		

	elif lable[i][2] == 1:
		downs.append([data[i],lable[i]])
	elif lable[i][3] == 1:
		rights.append([data[i],lable[i]])
	elif lable[i][4] == 1:
		forwards.append([data[i],lable[i]])
limit = min(len(lefts) , len(rights) , len(ups) , len(downs))

print(len(lefts) , len(rights) ,len(ups) , len(downs), len(forwards))
ll = [ups,lefts,downs,rights,forwards]
full_data = []

for l in ll:

	l = l[-limit:]
	full_data  = full_data + l
final_data = []
final_label = []
random.shuffle(full_data)

for dat in full_data:
	final_data.append(dat[0])
	final_label.append(dat[1])


np.save(file_name,final_data)
np.save(file_name2,final_label)



