import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split

# Define your custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Assuming you have images in a tensor 'images' and labels in a tensor 'one_hot_labels'
# You may need to reshape your labels depending on your data
# one_hot_labels = one_hot_labels.view(-1, num_classes)

# Split the dataset into training and validation sets
file_name = "test_data-9-imgs.npy"
file_name2 ="test_data-9-keys.npy"
data = np.load(file_name , allow_pickle=True)
labels = np.load(file_name2 , allow_pickle=True)
images = data
one_hot_labels = torch.FloatTensor(labels)


images_train, images_val, labels_train, labels_val = train_test_split(images, one_hot_labels, test_size=0.2, random_state=42)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create custom datasets and dataloaders
train_dataset = CustomDataset(images_train, labels_train, transform=transform)
val_dataset = CustomDataset(images_val, labels_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define the model
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)  # Assuming num_classes is the number of classes in your classification task

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    if(epoch % 5 == 4):
        torch.save(model.state_dict(), f"epoch-{epoch}-resnet18_model-9.pth")

    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss}")

# Save the trained model if needed
torch.save(model.state_dict(), 'resnet18_model-8.pth')