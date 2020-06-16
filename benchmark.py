import copy
import numpy as np
import argparse
# ---------------------------------------------
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# ---------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--lr', help='learning rate', default= 0.001, type=float)
parser.add_argument('--bsize', help='batch size', default= 32, type=int)
parser.add_argument('--testbsize', help='test batch size', default= 1000, type=int)
parser.add_argument('--epochs', help='train epochs', default= 50, type=int) 
arg = parser.parse_args()

device ='cuda' if torch.cuda.is_available() else 'cpu'

args = {
    'batch_size': arg.bsize,
    'test_batch_size': arg.testbsize,
    'lr': arg.lr,
    'log_interval': 100,
    'epochs': arg.epochs,
}
torch.manual_seed(1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    model = Net().to(device)

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True, num_workers=4)
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    test_dataloader = DataLoader(test_data, batch_size=args['test_batch_size'], shuffle=True, num_workers=4)

    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train_loss_list = []     
    test_accuracy_list = []
    test_loss_list = []
    
    for epoch in range(args['epochs']):
        model.train()
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss.item()))
            batch_loss.append(loss.item())
        train_loss_list.append(np.average(batch_loss))

        model.eval()
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                test_loss += criterion(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            test_loss /= len(test_dataloader)

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        
        test_accuracy_list.append((100 * correct / total))
        test_loss_list.append(test_loss)
    torch.save(model.state_dict(), './model/benchmark.pkl')
    
    print(train_loss_list)
    print(test_accuracy_list)
    print(test_loss_list)
    
    