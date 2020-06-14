import numpy as np
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

def gradinet_average(weights):
    w_avg = copy.deepcopy(weights[0])
    for k in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k]
        w_avg[k] = torch.div(w_avg[k], len(weights))
    return w_avg


if __name__ == '__main__':
    global_model = Net().to(device)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_dataloader = DataLoader(test_data, batch_size=args['test_batch_size'], shuffle=True)

    workers_data = random_split(train_data, args['train_proportion'])
    workers_dataloader = []
    for i in range(args['workers']):
        workers_dataloader.append(DataLoader(workers_data[i], batch_size=args['batch_size'], shuffle=True))

    global_train_loss = []
    global_test_loss = []
    global_test_accuracy = []

    for g_epoch in range(args['global_epochs']):
        global_model.train()

        concurrent_workers = np.random.randint(1,args['workers']+1)
        local_idx = np.random.choice(np.arange(args['workers']), concurrent_workers, replace=False)

        local_weight_list = []
        local_loss_list = []

        for idx in local_idx:
            local_model = copy.deepcopy(global_model).to(device)
            local_gradiett, local_loss = local_update(workers_dataloader[idx], local_model, idx)

            local_weight_list.append(local_weight)
            local_loss_list.append(local_loss)

        # modify algorithm with quantity of local data
        global_weight = federated_average(local_weight_list)
        global_model.load_state_dict(global_weight)

        avg_loss = np.average(local_loss_list)
        global_train_loss.append(avg_loss)
        print('Round {:3d},global average loss {:.3f}'.format(g_epoch, avg_loss))

        # eval -------------------------------------------
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

        global_test_accuracy.append(100. * correct / len(test_dataloader.dataset))
        global_test_loss.append(test_loss)

    print(global_test_accuracy)
    print(global_test_loss)
    print(global_train_loss)