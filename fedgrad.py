import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split

import numpy as np
import syft as sy
import copy

hook = sy.TorchHook(torch) # hook PyTorch to PySyft

args = {
    'batch_size' : 64,
    'test_batch_size' : 1000,
    'lr' : 0.01,
    'log_interval' : 100,
    'epochs' : 10,
    'workers' : 16,
}

worker_list = []
for i in range(args['workers']):
    worker_list.append(sy.VirtualWorker(hook, id=str(i)))

device ='cuda' if torch.cuda.is_available() else 'cpu'
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

test_data = datasets.MNIST('./data', train=False, transform = transform)

train_data = datasets.MNIST('./data', train=True, transform = transform)

train_proportion = [3750,3750,3750,3750,3750,3750,3750,3750,3750,3750,3750,3750,3750,3750,3750,3750]

workers_data = random_split(train_data, train_proportion)


federated_data = []


for i in range(args['workers']):
    idx_train = workers_data[i].indices
    train_inputs = train_data.data[idx_train].unsqueeze(1).float()
    train_targets = train_data.targets[idx_train]
    federated_data.append(sy.BaseDataset(train_inputs, train_targets).send(worker_list[i]))


federated_dataset = sy.FederatedDataset(federated_data)

federated_dataloader = sy.FederatedDataLoader(federated_dataset, shuffle=True, batch_size=args['batch_size'])

test_loader = DataLoader(test_data, shuffle=True, batch_size=args['test_batch_size'] )



def train(model, train_loader, optimizer, epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        model.send(data.location)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        model.get()

        if batch_idx % args['log_interval'] == 0:

            loss = loss.get()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * args['batch_size'],
                    len(train_loader) * args['batch_size'],
                    100. * batch_idx / len(train_loader),
                    loss.item()
                )
            )

def test(model, test_loader):

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'])
    epoch_loss = []

    for epoch in range(args['epochs']):
        train(model, federated_dataloader, optimizer, epoch)
        test(model, test_loader)


