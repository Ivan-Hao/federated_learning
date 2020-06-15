import numpy as np
import copy
import argparse
#------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split
#---------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--workers', help='the number of workers', default=4 ,type=int)
parser.add_argument('--lr', help='learning rate', default= 0.001, type=float)
parser.add_argument('--bsize', help='batch size', default= 32, type=int)
parser.add_argument('--testbsize', help='test batch size', default= 1000, type=int)
parser.add_argument('--gepochs', help='global train epochs', default= 40, type=int)
parser.add_argument('--lepochs', help='local train epochs', default= 1, type=int)
parser.add_argument('--lbtimes', help='local update batch times', default= 4, type=int)
parser.add_argument('--iid', help='i.i.d dataset ', default= 1, type=int)      
arg = parser.parse_args()

args = {
    'batch_size': arg.bsize,
    'test_batch_size': arg.testbsize,
    'lr': arg.lr,
    'global_epochs': arg.gepochs,
    'local_epochs': arg.lepochs,
    'workers': arg.workers,
    'train_proportion' : [50000//arg.workers] * arg.workers,
    'iid' : bool(arg.iid),
    'lbt' : arg.lbtimes
}


device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

def local_update(dataloader, model, optimizer, criterion, worker_id):

    model.train()
    i = 0 
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        
        i+=1
        if i == args['lbt']:
            return model.state_dict() ,loss.item()


if __name__ == '__main__':
    global_model = Net().to(device)
    global_weights = global_model.state_dict()
    global_model.train()
    
    criterion = nn.CrossEntropyLoss()
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train_data.targets = torch.Tensor(train_data.targets).long()
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


    local_model = []
    for i in range(args['workers']):
        local = Net().to(device)
        local.load_state_dict(copy.deepcopy(global_weights))
        local_model.append(local)

    optimizers = []
    for i in range(args['workers']):    
        optimizers.append(optim.SGD(local_model[i].parameters(), lr=args['lr'], momentum=0.9))

    global_train_loss = []
    global_test_accuracy = []
    global_test_loss = []

    for g_epoch in range(args['global_epochs']):

        concurrent_workers = np.random.randint(1,args['workers']+1)
        local_idx = np.random.choice(np.arange(args['workers']), concurrent_workers, replace=False)
        local_loss_list = []
        for i in range(args['local_epochs']):
            for j in range(len(workers_dataloader[0])//args['lbt']):
                for idx in local_idx:
                    local_weights, local_loss = local_update(workers_dataloader[idx], local_model[idx], optimizers[idx], criterion, idx)
                    local_loss_list.append(local_loss)
                    global_model.load_state_dict(copy.deepcopy(local_weights))
                    for l in local_model:
                        l.load_state_dict(copy.deepcopy(global_model.state_dict()))
                    
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
    torch.save(global_model.state_dict(), './model/grad'+str(args['workers'])+str(args['iid'])+'.pkl')

    print(global_test_accuracy)
    print(global_test_loss)
    print(global_train_loss)