import copy
import numpy as np
import argparse
# ---------------------------------------------
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split
# ----------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--workers', help='the number of workers', default=4 ,type=int)
parser.add_argument('--lr', help='learning rate', default= 0.001, type=float)
parser.add_argument('--bsize', help='batch size', default= 32, type=int)
parser.add_argument('--testbsize', help='test batch size', default= 1000, type=int)
parser.add_argument('--gepochs', help='global train epochs', default= 40, type=int)
parser.add_argument('--lepochs', help='local train epochs', default= 1, type=int)
parser.add_argument('--iid', help='i.i.d dataset ', default= 1, type=int)    
arg = parser.parse_args()

device ='cuda' if torch.cuda.is_available() else 'cpu'

args = {
    'batch_size': arg.bsize,
    'test_batch_size': arg.testbsize,
    'lr': arg.lr,
    'local_update_interval': 100,
    'global_epochs': arg.gepochs,
    'local_epochs': arg.lepochs,
    'workers': arg.workers,
    'train_proportion' : [50000//arg.workers] * arg.workers,
    'iid' : bool(arg.iid)
}

torch.manual_seed(1)
np.random.seed(1)
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


def federated_average(weights):
    w_avg = copy.deepcopy(weights[0])
    for k in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k]
        w_avg[k] = torch.div(w_avg[k], len(weights))
    return w_avg

def local_update(dataloader, model, criterion, worker_id):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9)
    epoch_loss = []
    for l_epoch in range(args['local_epochs']):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            model.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()

            optimizer.step()
            if batch_idx % args['local_update_interval'] == 0:
                print('Local Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tWorker_id: {}'.format(
                    l_epoch, batch_idx * len(data), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item(), worker_id))
                batch_loss.append(loss.item())
    
        epoch_loss.append(np.average(batch_loss))

    return model.state_dict(), np.average(epoch_loss)


if __name__ == '__main__':
    global_model = Net().to(device)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train_data.targets = torch.Tensor(train_data.targets).long() # = = list -> tensor wtf == 
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    test_dataloader = DataLoader(test_data, batch_size=args['test_batch_size'], shuffle=True)
    
    workers_dataloader = []
    
    if args['iid'] == True:
        workers_data = random_split(train_data, args['train_proportion'])
        for i in range(args['workers']):
            workers_dataloader.append(DataLoader(workers_data[i], batch_size=args['batch_size'], shuffle=True))
    else: # case for 5,10 workers
        if args['workers'] == 10:
            for i in range(args['workers']): #10
                subset = Subset(train_data,torch.where(train_data.targets == i)[0])
                workers_dataloader.append(DataLoader(subset , batch_size=args['batch_size'], shuffle=True))
        else:
            for i in range(args['workers']): # 5 
                idx1 = torch.where(train_data.targets == i*2)[0]
                idx2 = torch.where(train_data.targets == i*2+1)[0]
                idx = torch.cat((idx1,idx2),0)
                subset = Subset(train_data,idx)
                workers_dataloader.append(DataLoader(subset , batch_size=args['batch_size'], shuffle=True))

    global_train_loss = []
    global_test_accuracy = []
    global_test_loss = []

    for g_epoch in range(args['global_epochs']):
        global_model.train()

        concurrent_workers = np.random.randint(1,args['workers']+1)
        local_idx = np.random.choice(np.arange(args['workers']), concurrent_workers, replace=False)

        local_weight_list = []
        local_loss_list = []

        for idx in local_idx:
            local_model = copy.deepcopy(global_model).to(device)
            local_weight, local_loss = local_update(workers_dataloader[idx], local_model, criterion, idx)

            local_weight_list.append(copy.deepcopy(local_weight))
            local_loss_list.append(copy.deepcopy(local_loss))

        # modify algorithm with quantity of local data
        global_weight = federated_average(local_weight_list)
        global_model.load_state_dict(global_weight)

        avg_loss = np.average(local_loss_list)
        global_train_loss.append(avg_loss)
        print('Round {:3d},global average loss {:.3f}'.format(g_epoch, avg_loss))

        # eval -------------------------------------------
        global_model.eval()
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = global_model(data)
                test_loss += criterion(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            test_loss /= len(test_dataloader)

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

        global_test_accuracy.append((100 * correct / total))
        global_test_loss.append(test_loss)
    torch.save(global_model.state_dict(), './model/avg'+str(args['workers'])+str(args['iid'])+'.pkl')
    print(global_test_accuracy)
    print(global_test_loss)
    print(global_train_loss)



