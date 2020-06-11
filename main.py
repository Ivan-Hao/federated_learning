import torch
import torch.nn.functional as F
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split
import numpy as np
import logging

# import Pysyft to help us to simulate federated leraning
import syft as sy


hook = sy.TorchHook(torch) # hook PyTorch to PySyft

# define the args
args = {
    'batch_size' : 64,
    'test_batch_size' : 1000,
    'lr' : 0.01,
    'log_interval' : 100,
    'epochs' : 10,
    'workers' : 2,
}

worker_list = []
for i in range(args['workers']):
    worker_list.append(sy.VirtualWorker(hook, id=str(i)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# We should modify the model 
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=64*12*12, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
        )
    
    def forward(self, x):
        #print(x)
        #print(x.shape)
        x = self.conv(x)
        #print(x)
        x = F.max_pool2d(x,2)
        x = x.view(-1, 64*12*12)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


test_data = datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

test_len = test_data.__len__()

train_data = datasets.MNIST('./data', train=True,transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

train_len = train_data.__len__()

'''
proportion = np.random.dirichlet(np.ones(args['workers'],dtype=np.float64),size=1)
while proportion.sum() != 1:
    proportion = np.random.dirichlet(np.ones(args['workers'],dtype=np.float64),size=1)

train_data_proportion = np.around(train_len*proportion)
test_data_proportion = np.around(test_len*proportion)
'''
fix_train = [30000,30000]

works_train_data = random_split(train_data,fix_train)#random_split(train_data, train_data_proportion)


federate_data_train = [] 


for i in range(args['workers']):
    idx_train = works_train_data[i].indices
    train_inputs = train_data.data.unsqueeze(1).float()[idx_train]
    train_labels = train_data.targets[idx_train]
    federate_data_train.append(sy.BaseDataset(train_inputs, train_labels).send(worker_list[i]))


federated_train_dataset = sy.FederatedDataset(federate_data_train)

federated_train_loader = sy.FederatedDataLoader(federated_train_dataset, shuffle=True, batch_size=args['batch_size'])

test_loader = torch.utils.data.DataLoader(test_data, shuffle=False,batch_size=args['test_batch_size'] )

#print(train_data.data.shape)
#print(train_data.data.unsqueeze(1).shape)
#exit()	

# we can look at the data, it is actually pointer tensors
for images,labels in federated_train_loader:
    print(images.shape) # batch of images pointers
    print(labels.shape) # batch of image labels pointers
    
    print(len(images)) # len function works on pointers as well
    print(len(labels)) # we can see both are same, no of images as well as their labels
    break
for images,labels in test_loader:
    print(images.shape) # batch of images pointers
    print(labels.shape) # batch of image labels pointers
    
    print(len(images)) # len function works on pointers as well
    print(len(labels)) # we can see both are same, no of images as well as their labels
    break

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    # iterate over federated data
    for batch_idx, (data, target) in enumerate(train_loader):

        # send the model to the remote location 
        model = model.send(data.location)

        # the same torch code that we are use to
        data, target = data.to(device), target.to(device)
        #data, target = data.unsqueeze(1).float().to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # this loss is a ptr to the tensor loss 
        # at the remote location
        loss = F.nll_loss(output, target)

        # call backward() on the loss ptr,
        # that will send the command to call
        # backward on the actual loss tensor
        # present on the remote machine
        loss.backward()

        optimizer.step()

        # get back the updated model
        model.get()

        if batch_idx % args['log_interval'] == 0:

            # a thing to note is the variable loss was
            # also created at remote worker, so we need to
            # explicitly get it back
            loss = loss.get()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, 
                    batch_idx * args['batch_size'], # no of images done
                    len(train_loader) * args['batch_size'], # total images left
                    100. * batch_idx / len(train_loader), 
                    loss.item()
                )
            )

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #data, target = data.unsqueeze(1).float().to(device), target.to(device)
            #print(data.shape, target.shape)
            output = model(data)

            # add losses together
            test_loss += F.nll_loss(output, target, reduction='sum').item() 
            #test_loss += F.mse_loss(output, target, reduction='sum').item()

            # get the index of the max probability class
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=args['lr'])

logging.info("Starting training !!")

for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, federated_train_loader, optimizer, epoch)
        test(model, device, test_loader)
    
# thats all we need to do XD
