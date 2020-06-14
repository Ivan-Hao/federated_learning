import copy
import numpy as np
import argparse
# ---------------------------------------------
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split
# ----------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('--lr', help='learning rate', default= 0.01, type=float)
parser.add_argument('--bsize', help='batch size', default= 64, type=int)
parser.add_argument('--testbsize', help='test batch size', default= 1000, type=int)
parser.add_argument('--epochs', help='train epochs', default= 20, type=int) 
arg = parser.parse_args()

device ='cuda' if torch.cuda.is_available() else 'cpu'

args = {
    'batch_size': arg.bsize,
    'test_batch_size': arg.testbsize,
    'lr': arg.lr,
    'log_interval': 100,
    'epochs': arg.epochs,
}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 5, 1)
        self.fc1 = nn.Linear(288, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

if __name__ == '__main__':
    model = Net().to(device)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_dataloader = DataLoader(test_data, batch_size=args['test_batch_size'], shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=args['lr'])

    test_loss_list = []
    test_accuracy_list = []
    train_loss_list = [] 

    for epoch in range(args['epochs']):
        model.train()
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss.item()))
                batch_loss.append(loss.item())
        train_loss_list.append(np.average(batch_loss))

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_dataloader.dataset)
        test_loss_list.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_dataloader.dataset),
            100. * correct / len(test_dataloader.dataset)))
        
        test_accuracy_list.append(100. * correct / len(test_dataloader.dataset))
    
    print(test_loss_list)
    print(test_accuracy_list)
    print(train_loss_list)