import numpy as np
import syft as sy
import copy
import argparse
#------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split
#---------------------------------------------------

hook = sy.TorchHook(torch) # hook PyTorch to PySyft

parser = argparse.ArgumentParser()
parser.add_argument('--workers', help='the number of workers', default=4 ,type=int)
parser.add_argument('--lr', help='learning rate', default= 0.01, type=float)
parser.add_argument('--bsize', help='batch size', default= 64, type=int)
parser.add_argument('--testbsize', help='test batch size', default= 1000, type=int)
parser.add_argument('--gepochs', help='global train epochs', default= 20, type=int)
parser.add_argument('--lepochs', help='local train epochs', default= 1, type=int)    
arg = parser.parse_args()

args = {
    'batch_size': arg.bsize,
    'test_batch_size': arg.testbsize,
    'lr': arg.lr,
    'local_update_interval': 100,
    'global_epochs': arg.gepochs,
    'local_epochs': arg.lepochs,
    'workers': arg.workers,
    'train_proportion' : [60000//arg.workers] * arg.workers
}

worker_list = []
for i in range(args['workers']):
    worker_list.append(sy.VirtualWorker(hook, id=str(i)))

device ='cuda' if torch.cuda.is_available() else 'cpu'

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

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
test_data = datasets.MNIST('./data', train=False, transform = transform)
train_data = datasets.MNIST('./data', train=True, transform = transform)


train_data.data = train_data.data.unsqueeze(1).float()
workers_data = random_split(train_data, args['train_proportion'])

federated_data = []

for i in range(args['workers']):
    idx_train = workers_data[i].indices
    train_inputs = train_data.data[idx_train]
    train_targets = train_data.targets[idx_train]
    federated_data.append(sy.BaseDataset(train_inputs, train_targets).send(worker_list[i]))

federated_dataset = sy.FederatedDataset(federated_data)
federated_dataloader = sy.FederatedDataLoader(federated_dataset, shuffle=True, batch_size=args['batch_size'])

#federated_dataloader = sy.FederatedDataLoader(train_data.federate(worker_list), batch_size=args['batch_size'], shuffle=True)

test_loader = DataLoader(test_data, shuffle=True, batch_size=args['test_batch_size'] )

def train(model, train_loader, optimizer, epoch):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.location)

        model.send(data.location)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        model.get()

        if batch_idx % args['local_update_interval'] == 0:
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

    for epoch in range(args['global_epochs']):
        train(model, federated_dataloader, optimizer, epoch)
        test(model, test_loader)


