import os
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import pickle

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


covid_path = 'C:/Users/crisr/Desktop/COVID/COVID-19_Radiography_Dataset/COVID'
healthy_path = 'C:/Users/crisr/Desktop/COVID/COVID-19_Radiography_Dataset/Normal'

losses = {'train': list(), 'validacion': list(), 'test': list()}

class RadDataset(Dataset):
	def __init__(self, file_path):
		#List of radiography files:
		self.radfiles = os.listdir(file_path)   
		#print(self.radfiles)

		self.file_path = file_path
	def __len__(self):
		return len(self.radfiles)
	def __getitem__(self, idx):
		f_path = self.file_path + '/' +  self.radfiles[idx] 
		# Load the image:
		x = Image.open(f_path)
		x = x.resize((128,128),Image.ANTIALIAS)
		if x.mode == 'L':
			x = x.convert('RGB')
		tensor = transforms.ToTensor()
		x=tensor(x)
		without_extra_slash = os.path.normpath(self.file_path)
		last_part = os.path.basename(without_extra_slash)
		if last_part=='COVID':
			y=1 	#COVID PATIENT, y=1
		else:
			y=0		#HEALTHY PATIENT, y=0
		return x,y


covid_set = RadDataset(covid_path)
healthy_set = RadDataset(healthy_path)


#Number of radiographies inside the chosen folder:
#print(covid_set.__len__())

#Get one data item:
#healthy_set.__getitem__(1)

#Show an image with its index:
#healthy_trainset.__getitem__(120)[0].show()


#Fuse sets:
cat_set = ConcatDataset([covid_set, healthy_set])

#Divide dataset in train, validation and test sets:
train_size = int(round(0.8 * len(cat_set)))
val_size = int(round(0.1* len(cat_set)))
test_size = int(round(0.1* len(cat_set)))


train_set, val_set, test_set = random_split(cat_set,[train_size, val_size, test_size])

batch_size=3
train_loader = DataLoader(train_set, batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)



#CNN:

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.max_pool = nn.MaxPool2d((2,2))
		self.fc1 = nn.Linear(256*batch_size, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 1)

		self.conv1 = nn.Conv2d(3, 32, (3,3))
		self.conv2 = nn.Conv2d(32, 32, (3,3))
		self.conv3 = nn.Conv2d(32, 64, (3,3))
		self.conv4 = nn.Conv2d(64, 64, (3,3))

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.max_pool(x)

		x = F.relu(self.conv2(x))
		x = self.max_pool(x)

		x = F.relu(self.conv2(x))
		x = self.max_pool(x)

		x = F.relu(self.conv3(x))
		x = self.max_pool(x)

		x = F.relu(self.conv4(x))
		x = self.max_pool(x)

		x = x.flatten()

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = torch.sigmoid(self.fc3(x))

		return x


model=CNN()
model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#First trainig loop (Adam):
epochs=14
for epoch in range(epochs):

	#TRAINING: 
	print("Training... \n")
	training_loss = 0.0
	total = 0
	model.train()
	for i, data in enumerate(train_loader, 0):
		x,y = data

		total += 1

		# zero the parameter gradients
		optimizer.zero_grad()

		x = x.to(device)
		y = y.to(device)

		# forward + backward + optimize
		output = model(x)
		y=y.float()
		loss = criterion(output,y)
		loss.backward()
		optimizer.step()


		# print statistics
		training_loss += loss.item()
	
	training_loss = training_loss/total
	losses['train'].append(training_loss)
		
	
	#VALIDATION
	model.eval()  #Prepare model for validation/test
	print("Validating... \n")
	val_loss=0.0
	total = 0
	for j, data in enumerate(val_loader, 0):
		x,y = data

		total += 1

		x = x.to(device)
		y = y.to(device)

		output = model(x)  # forward
		y=y.float()
		loss = criterion(output, y) 
		val_loss += loss.item()

	val_loss  = val_loss/total
	losses['validacion'].append(val_loss)

	print("Epoch: " + str(epoch+1) + ", Training loss: " + str(training_loss) + ", Validation loss: " + str(val_loss))

#Second trainig loop (fining with SGD):
optimizer = optim.SGD(model.parameters(), lr=1e-5)
epochs=6
for epoch in range(epochs):

	#TRAINING: 
	print("Training... \n")
	training_loss = 0.0
	total = 0
	model.train()
	for i, data in enumerate(train_loader, 0):
		x,y = data

		total += 1

		# zero the parameter gradients
		optimizer.zero_grad()

		x = x.to(device)
		y = y.to(device)

		# forward + backward + optimize
		output = model(x)
		y=y.float()
		loss = criterion(output,y)
		loss.backward()
		optimizer.step()


		# print statistics
		training_loss += loss.item()
	
	training_loss = training_loss/total
	losses['train'].append(training_loss)
		
	
	#VALIDATION
	model.eval()  #Prepare model for validation/test
	print("Validating... \n")
	val_loss=0.0
	total = 0
	for j, data in enumerate(val_loader, 0):
		x,y = data

		total += 1

		x = x.to(device)
		y = y.to(device)

		output = model(x)  # forward
		y=y.float()
		loss = criterion(output, y) 
		val_loss += loss.item()

	val_loss  = val_loss/total
	losses['validacion'].append(val_loss)

	print("Epoch: " + str(epoch+1) + ", Training loss: " + str(training_loss) + ", Validation loss: " + str(val_loss))
print('Finished Training')


#Save trained model with pickle
filename = 'C:/Users/crisr/Desktop/COVID/COVID-19_Radiography_Dataset/covid.sav'
pickle.dump(model, open(filename, 'wb'))

# TEST
model.eval()  #Prepare model for validation/test
print("Testing... \n")
test_loss=0.0
yreal = list()
ypredicha = list()
total=0

for k,data in  enumerate(test_loader, 0):
	x,y=data
	total +=1
	x = x.to(device)
	y = y.to(device)
	with torch.no_grad():
		output = model(x)

	y=y.float()
	loss = criterion(output, y) 

	yreal.append(y.data.cpu().numpy())
	ypredicha.append(output.data.cpu().numpy())

	test_loss += loss.item()
test_loss  = test_loss/total
print("Test loss: " + str(test_loss))
losses['test'].append(test_loss)


losses['yreal'] = np.array([el.item() for a in yreal for el in a])
losses['ypredicha'] = np.array([el.item() for a in ypredicha for el in a])

#Save losses in a pickle:
filename = 'C:/Users/crisr/Desktop/COVID/COVID-19_Radiography_Dataset/losses_covid.pickle'
with open(filename, 'wb') as handle:
	pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Print test real values and its predictions:
print(losses['yreal'],losses['ypredicha'])
