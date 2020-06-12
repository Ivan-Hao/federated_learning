import copy
import numpy as np
# ---------------------------------------------
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split
# ----------------------------------------------


device ='cuda' if torch.cuda.is_available() else 'cpu'

args = {
    'batch_size':64,
    'test_batch_size':1000,
    'lr':0.01,
    'local_update_interval':100,
    'global_epochs':10,
    'local_epochs':3,
    'workers':2,

}


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

def federated_average(weights):
    w_avg = copy.deepcopy(weights[0])
    for k in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k]
        w_avg[k] = torch.div(w_avg[k], len(weights))
    return w_avg

def local_update(dataloader, model, worker_id):
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.5)

    epoch_loss = []
    for l_epoch in range(args['local_epochs']):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            model.zero_grad()
            output = model(data)

            loss = F.nll_loss(output, target)
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

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_dataloader = DataLoader(test_data, batch_size=args['test_batch_size'], shuffle=True)

    train_proportion = [30000,30000]
    workers_data = random_split(train_data, train_proportion)
    workers_dataloader = []
    for i in range(args['workers']):
        workers_dataloader.append(DataLoader(workers_data[i], batch_size=args['batch_size'], shuffle=True))

    global_train_loss = []
    global_test_loss = []


    for g_epoch in range(args['global_epochs']):
        global_model.train()

        concurrent_workers = np.random.randint(1,args['workers']+1)
        local_idx = np.random.choice(np.arange(args['workers']), concurrent_workers, replace=False)

        local_weight_list = []
        local_loss_list = []

        for idx in local_idx:
            local_model = copy.deepcopy(global_model).to(device)
            local_weight, local_loss = local_update(workers_dataloader[idx], local_model, idx)

            local_weight_list.append(local_weight)
            local_loss_list.append(local_loss)

        # modify algorithm with quantity of local data
        global_weight = federated_average(local_weight_list)
        global_model.load_state_dict(global_weight)

        avg_loss = np.average(local_loss_list)
        global_train_loss.append(avg_loss)
        print('Round {:3d},global average loss {:.3f}'.format(g_epoch, avg_loss))

        global_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                output = global_model(data)

                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_dataloader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_dataloader.dataset),
            100. * correct / len(test_dataloader.dataset)))






