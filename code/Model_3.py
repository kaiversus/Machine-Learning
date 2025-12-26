import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()

        self.conv3_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn3_1   = nn.BatchNorm2d(64)

        self.conv3_2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3_2   = nn.BatchNorm2d(128)

        self.conv3_3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_3   = nn.BatchNorm2d(256)

        self.conv5_1 = nn.Conv2d(3, 64, 5, padding=2)
        self.bn5_1   = nn.BatchNorm2d(64)

        self.conv5_2 = nn.Conv2d(64, 128, 5, padding=2)
        self.bn5_2   = nn.BatchNorm2d(128)

        self.conv5_3 = nn.Conv2d(128, 256, 5, padding=2)
        self.bn5_3   = nn.BatchNorm2d(256)

        self.conv3_4 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn3_4   = nn.BatchNorm2d(256)

        self.conv5_4 = nn.Conv2d(512, 256, 5, padding=2)
        self.bn5_4   = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv3_5 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_5   = nn.BatchNorm2d(256)

        self.conv3_6 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn3_6   = nn.BatchNorm2d(128)

        self.conv5_5 = nn.Conv2d(256, 256, 5, padding=2)
        self.bn5_5   = nn.BatchNorm2d(256)

        self.conv5_6 = nn.Conv2d(256, 128, 5, padding=2)
        self.bn5_6   = nn.BatchNorm2d(128)

        self.conv3_7 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn3_7   = nn.BatchNorm2d(128)

        self.conv5_7 = nn.Conv2d(256, 128, 5, padding=2)
        self.bn5_7   = nn.BatchNorm2d(128)

        self.conv3_8  = nn.Conv2d(128, 64, 3, padding=1)
        self.bn3_8    = nn.BatchNorm2d(64)

        self.conv3_9  = nn.Conv2d(64, 32, 3, padding=1)
        self.bn3_9    = nn.BatchNorm2d(32)

        self.conv3_10 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn3_10   = nn.BatchNorm2d(16)

        self.conv5_8  = nn.Conv2d(128, 64, 5, padding=2)
        self.bn5_8    = nn.BatchNorm2d(64)

        self.conv5_9  = nn.Conv2d(64, 32, 5, padding=2)
        self.bn5_9    = nn.BatchNorm2d(32)

        self.conv5_10 = nn.Conv2d(32, 16, 5, padding=2)
        self.bn5_10   = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(32 * 4 * 4, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):

        x3 = F.relu(self.bn3_1(self.conv3_1(x)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3)))
        x3 = F.relu(self.bn3_3(self.conv3_3(x3)))

        x5 = F.relu(self.bn5_1(self.conv5_1(x)))
        x5 = F.relu(self.bn5_2(self.conv5_2(x5)))
        x5 = F.relu(self.bn5_3(self.conv5_3(x5)))

        x = torch.cat([x3, x5], dim=1)

        x_main = F.relu(self.bn3_4(self.conv3_4(x)))
        x_skip = F.relu(self.bn5_4(self.conv5_4(x)))
        x = x_main + x_skip

        x = self.pool(x)

        x3 = F.relu(self.bn3_5(self.conv3_5(x)))
        x3 = F.relu(self.bn3_6(self.conv3_6(x3)))

        x5 = F.relu(self.bn5_5(self.conv5_5(x)))
        x5 = F.relu(self.bn5_6(self.conv5_6(x5)))

        x = torch.cat([x3, x5], dim=1)

        x_main = F.relu(self.bn3_7(self.conv3_7(x)))
        x_skip = F.relu(self.bn5_7(self.conv5_7(x)))
        x = x_main + x_skip

        x = self.pool(x)

        x3 = F.relu(self.bn3_8(self.conv3_8(x)))
        x3 = F.relu(self.bn3_9(self.conv3_9(x3)))

        x5 = F.relu(self.bn5_8(self.conv5_8(x)))
        x5 = F.relu(self.bn5_9(self.conv5_9(x5)))

        x = x3 + x5

        x_main = F.relu(self.bn3_10(self.conv3_10(x)))
        x_skip = F.relu(self.bn5_10(self.conv5_10(x)))

        x = torch.cat([x_main, x_skip], dim=1)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


net = Net().to(device)
from torchsummary import summary
net = net.to(device)
summary(net, (3, 32, 32))

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0

    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 200 == 199:
            print(f"[Epoch {epoch+1}, Batch {i+1}] "
                  f"loss: {running_loss / 200:.4f}")
            running_loss = 0.0

    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, '/content/checkpoint.pth')

print('Finished Training')


dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        inputs, labels = inputs.to(device), labels.to(device)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
