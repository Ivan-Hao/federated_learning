import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import logging

# import Pysyft to help us to simulate federated leraning
import syft as sy

# hook PyTorch to PySyft i.e. add extra functionalities to support Federated Learning
# and other private AI tools
hook = sy.TorchHook(torch) 

# we create two imaginary schools
westside_school = sy.VirtualWorker(hook, id="westside")
grapevine_high = sy.VirtualWorker(hook, id="grapevine")

# define the args
args = {
    'use_cuda' : True,
    'batch_size' : 64,
    'test_batch_size' : 1000,
    'lr' : 0.01,
    'log_interval' : 100,
    'epochs' : 10
}

# check to use GPU or not
use_cuda = args['use_cuda'] and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# create a simple CNN net
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
        x = self.conv(x)
        x = F.max_pool2d(x,2)
        x = x.view(-1, 64*12*12)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# Now we take the help of PySyft's awesome API to prepare the data for us and
# distribute for us across 2 workers ie. two schools
# normally we dont have to distribute data, data is already there at the site.
# We are doing this just to simulate federated learning.
# Below code looks just like torch code with just some minor changes. This is what's nice about PySyft.
federated_train_loader = sy.FederatedDataLoader(
    datasets.MNIST('./data', train=True, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    .federate((grapevine_high, westside_school)),
    batch_size=args['batch_size'], shuffle=True)

# test data remains with us locally
# this is the normal torch code to load test data from MNIST
# that we are all familiar with
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args['test_batch_size'], shuffle=True)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    # iterate over federated data
    for batch_idx, (data, target) in enumerate(train_loader):

        # send the model to the remote location 
        model = model.send(data.location)

        # the same torch code that we are use to
        data, target = data.to(device), target.to(device)
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
            output = model(data)

            # add losses together
            test_loss += F.nll_loss(output, target, reduction='sum').item() 

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
    