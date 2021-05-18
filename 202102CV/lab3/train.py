import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
from torch.utils.data import DataLoader
from dataset import trainset, testset, imshow
from model import LeNet5

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')

trainloader = DataLoader(trainset, batch_size=8, shuffle=True,)
testloader = DataLoader(testset, batch_size=8, shuffle=True)
classes = trainset.classes

model = LeNet5
model=model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# loop over the dataset multiple times
num_epoch = 10
for epoch in range(num_epoch):
    running_loss = 0.0
    for i, batch in enumerate(trainloader, 0):
        # get the images; batch is a list of [images, labels]
        images, labels = batch
        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()  # zero the parameter gradients

        # get prediction
        outputs = model(images)
        # compute loss
        loss = loss_fn(outputs, labels)
        # reduce loss
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:  # print every 500 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

print('Finished Training')

def accuracy(model, data_loader):
    correct, total = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
            images=images.to(device)
            labels=labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


train_acc = accuracy(model, trainloader)
test_acc = accuracy(model, testloader)

print('Accuracy on the train set: %f %%' % (100 * train_acc))
print('Accuracy on the test set: %f %%' % (100 * test_acc))

torch.save(LeNet5.state_dict(), 'model.pth')
print('Model saved.')
