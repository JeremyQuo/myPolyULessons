import torch
import torchvision
import numpy as np
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms


torch.manual_seed(28)
torch.cuda.manual_seed(28)
# set device
device = torch.device('cpu')

class ass3(nn.Module):
    def __init__(self, in_channels=784, num_classes=10):
        super(ass3, self).__init__()
        self.fc1 = nn.Linear(in_channels,32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x= self.fc2(x)
        return x

# Hyper-parameters

num_classes = 10
learning_rate = 1e-4
batch_size = 32
num_epochs = 5

# Loss and optimizer
model = ass3(in_channels=784,num_classes = num_classes)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#print parameter
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

#load dataset
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
vis_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)


for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        # forward
        data=data.reshape(-1,28*28)
        scores = model(data)
        targets=nn.functional.one_hot(targets,num_classes).float()
        loss = criterion(scores, targets)

        #print(epoch,loss)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 99:  # print every 200 mini-batches
            print('[epoch%d, mini_batch%5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 100))
            running_loss = 0.0

# Check accuracy on training & test to see how good our model
torch.save(model.state_dict(), 'ass3')



def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.reshape(-1, 28 * 28)
            test_scores = model(x)
            _, predictions = test_scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()

    return test_scores


# check accuracy in test data
#
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
#
