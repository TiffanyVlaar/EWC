
import numpy as np
import matplotlib.cm as cm
import numpy.random as npr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

torch.set_default_tensor_type('torch.cuda.DoubleTensor')

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(2)

print(f"Running on {device}.")

mnist_trainold = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
loader_train = DataLoader(mnist_trainold, batch_size=len(mnist_trainold), shuffle=True)

mnist_testold = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
loader_test = DataLoader(mnist_testold, batch_size=len(mnist_testold), shuffle=True)


for i,data in enumerate(loader_train):
    xtrainold=data[0].view(-1,784)
    ytrainold=data[1] 

for i,data in enumerate(loader_test):
    xtestold=data[0].view(-1,784)
    ytestold=data[1] 


idx1 = np.arange(784)
np.random.shuffle(idx1)

idx2 = np.arange(784)
np.random.shuffle(idx2)


class PermutedMNIST(Dataset):
    
    def __init__(self,x,y,idx):
        super().__init__()
        self.y = y
        self.x = x[:,idx]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]


batchsize = 256

mnist_train1 = PermutedMNIST(xtrainold, ytrainold,idx1)
dataloader_train1 = DataLoader(mnist_train1, batch_size=batchsize, shuffle=True)

mnist_test1 = PermutedMNIST(xtestold, ytestold,idx1)
dataloader_test1 = DataLoader(mnist_test1, batch_size=len(mnist_test1), shuffle=True)

mnist_train2 = PermutedMNIST(xtrainold, ytrainold,idx2)
dataloader_train2 = DataLoader(mnist_train2, batch_size=batchsize, shuffle=True)

mnist_test2 = PermutedMNIST(xtestold, ytestold,idx2)
dataloader_test2 = DataLoader(mnist_test2, batch_size=len(mnist_test2), shuffle=True)


for data_test in enumerate(dataloader_test1):
    x_test1 = data_test[1][0].view(-1,784).double().cuda()
    y_test1 = data_test[1][1]

for data_test in enumerate(dataloader_test2):
    x_test2 = data_test[1][0].view(-1,784).double().cuda()
    y_test2 = data_test[1][1]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.big_pass = nn.Sequential(nn.Linear(784,500),
                                      nn.ReLU(), 
                                      nn.Linear(500,200),
                                      nn.ReLU(), 
                                      nn.Linear(200,10)
                                     )

    def forward(self, x):
        out = self.big_pass(x)
        return out 

def accuracy_Nclass(out,y):
    diff = np.count_nonzero(np.argmax(out,axis=1)-y)
    return (1-(diff/np.size(y)))*100


# # Standard SGD
num_epochs = 50
total_steps = len(dataloader_train1)

h = 0.01

RES1 = []
NN = Net()
NN = NN.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(NN.parameters(),lr=h)

for epoch in range(num_epochs): 
    for i,data in enumerate(dataloader_train1):
         # Load in the training datapoints
        x=data[0].view(-1,784).double().cuda()
        y=data[1].long().cuda()

        output = NN(x)
        loss = criterion(output,y) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        # Evaluate how the neural network is doing and store the results
        if (epoch+1) % 1 == 0 and (i+1) % int(len(mnist_train1)/batchsize) == 0: 
            
            acc = accuracy_Nclass(output.cpu().detach().numpy(),y.cpu().detach().numpy())
            outputtest = NN(x_test1)
            loss_test = criterion(outputtest.cpu(),y_test1)
            acc_test = accuracy_Nclass(outputtest.cpu().detach().numpy(),y_test1.detach().numpy())
            print(f'epoch {epoch}/{num_epochs}, step {i+1}/{total_steps}, accuracy test {acc_test} %, lost test, {loss_test.item()}')

                # Store the results
            RES1 += [[epoch ,loss_test.item(),acc_test, loss.item(), acc]] 

total_steps = len(dataloader_train2)
RES2 = []

for epoch in range(num_epochs): 
    for i,data in enumerate(dataloader_train2):
        x=data[0].view(-1,784).double().cuda()
        y=data[1].long().cuda()

        output = NN(x)
        loss = criterion(output,y) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (epoch+1) % 1 == 0 and (i+1) % int(len(mnist_train2)/batchsize) == 0: 
            acc = accuracy_Nclass(output.cpu().detach().numpy(),y.cpu().detach().numpy())
            
            outputtest1 = NN(x_test1)
            loss_test1 = criterion(outputtest1.cpu(),y_test1)
            acc_test1 = accuracy_Nclass(outputtest1.cpu().detach().numpy(),y_test1.detach().numpy())
            
            outputtest2 = NN(x_test2)
            loss_test2 = criterion(outputtest2.cpu(),y_test2)
            acc_test2 = accuracy_Nclass(outputtest2.cpu().detach().numpy(),y_test2.detach().numpy())
            
            print(f'epoch {epoch}/{num_epochs}, step {i+1}/{total_steps}, accuracy test1 {acc_test1} %, lost test1, {loss_test1.item()}')
            print(f'accuracy test2 {acc_test2} %, lost test2, {loss_test2.item()}')
            
                # Store the results
            RES1 += [[epoch+50, loss_test1.item(),acc_test1,loss.item(), acc]] 
            RES2 += [[epoch+50, loss_test2.item(),acc_test2]] 

RES1 = np.vstack(RES1)
RES2 = np.vstack(RES2)

plt.figure(figsize=[8,5]) # Increase the size of the plots
plt.rcParams.update({'font.size': 18})

plt.plot(RES1[:,0],RES1[:,2],label="Task A")
plt.plot(RES2[:,0],RES2[:,2],label="Task B")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("Original.pdf")

# # EWC
num_epochs = 50
total_steps = len(dataloader_train1)

h = 0.01

RES1_EWC = []
NN_EWC = Net()
NN_EWC.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(NN_EWC.parameters(),lr=h)

for epoch in range(num_epochs): 
    for i,data in enumerate(dataloader_train1):
        x=data[0].view(-1,784).double().cuda()
        y=data[1].long().cuda()

        output = NN_EWC(x)
        loss = criterion(output,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 1 == 0 and (i+1) % int(len(mnist_train1)/batchsize) == 0: 
            
            acc = accuracy_Nclass(output.cpu().detach().numpy(),y.cpu().detach().numpy())
            outputtest = NN_EWC(x_test1)
            loss_test = criterion(outputtest.cpu(),y_test1)
            acc_test = accuracy_Nclass(outputtest.cpu().detach().numpy(),y_test1.detach().numpy())
            print(f'epoch {epoch}/{num_epochs}, step {i+1}/{total_steps}, accuracy test {acc_test} %, lost test, {loss_test.item()}')

                # Store the results
            RES1_EWC += [[epoch ,loss_test.item(),acc_test, loss.item(), acc]] 


torch.save(NN_EWC.state_dict(), 'test1')

NN_EWC1 = Net()
NN_EWC1.load_state_dict(torch.load('test1'))
NN_EWC1.eval()

weights_taskA = []

for param in NN_EWC1.parameters():
    weights_taskA.append(param.clone())

dataloader_train1_full = DataLoader(mnist_train1, batch_size=32,shuffle=True)

Fisher = []
loglik = []
for i,data in enumerate(dataloader_train1_full):
    x=data[0].view(32,-1).double().cuda()
    y=data[1].long().cuda()
    x = Variable(x) 
    y = Variable(y)
    loglik.append(F.log_softmax(NN_EWC(x), dim=1)[range(32),y.data])
    if len(loglik) >= 2048//32:
        break

loglik = torch.cat(loglik).unbind()
loglik_grads = zip(*[torch.autograd.grad(l,NN_EWC.parameters(),retain_graph=True) for i,l in enumerate(loglik,1)])    
loglik_grads = [torch.stack(gs) for gs in loglik_grads]
fisher = [(g**2).mean(0) for g in loglik_grads]

total_steps = len(dataloader_train2)
RES1_EWC = []
RES2_EWC = []
importance = 100 
h = 0.01
for epoch in range(num_epochs): 
    for i,data in enumerate(dataloader_train2):
        x=data[0].view(-1,784).double().cuda()
        y=data[1].long().cuda()

        output = NN_EWC1(x)
        lossB = criterion(output,y)

        loss_EWC = 0
        for m in range(3):
            loss_EWC += torch.sum(fisher[2*m].reshape(-1,1)*((NN_EWC1.big_pass[2*m].weight-weights_taskA[2*m])**2).reshape(-1,1))
            loss_EWC += torch.sum(fisher[2*m+1].reshape(-1,1)*((NN_EWC1.big_pass[2*m].bias-weights_taskA[2*m+1])**2).reshape(-1,1))
            
        loss = lossB+importance*loss_EWC
        
        dL1 = torch.autograd.grad(loss,NN_EWC1.big_pass[0].weight,create_graph=True)[0]
        dL2 = torch.autograd.grad(loss,NN_EWC1.big_pass[2].weight,create_graph=True)[0]
        dL3 = torch.autograd.grad(loss,NN_EWC1.big_pass[4].weight,create_graph=True)[0]
        dLb1 = torch.autograd.grad(loss,NN_EWC1.big_pass[0].bias,create_graph=True)[0]
        dLb2 = torch.autograd.grad(loss,NN_EWC1.big_pass[2].bias,create_graph=True)[0]
        dLb3 = torch.autograd.grad(loss,NN_EWC1.big_pass[4].bias)[0]
        
        NN_EWC1.big_pass[0].weight = torch.nn.Parameter(NN_EWC1.big_pass[0].weight - h*dL1)
        NN_EWC1.big_pass[2].weight = torch.nn.Parameter(NN_EWC1.big_pass[2].weight - h*dL2)
        NN_EWC1.big_pass[4].weight = torch.nn.Parameter(NN_EWC1.big_pass[4].weight - h*dL3)
        NN_EWC1.big_pass[0].bias = torch.nn.Parameter(NN_EWC1.big_pass[0].bias - h*dLb1)
        NN_EWC1.big_pass[2].bias = torch.nn.Parameter(NN_EWC1.big_pass[2].bias - h*dLb2)
        NN_EWC1.big_pass[4].bias = torch.nn.Parameter(NN_EWC1.big_pass[4].bias - h*dLb3)

        
        if (epoch+1) % 1 == 0 and (i+1) % int(len(mnist_train2)/batchsize) == 0: 
            acc = accuracy_Nclass(output.cpu().detach().numpy(),y.cpu().detach().numpy())
            
            outputtest1 = NN_EWC1(x_test1)
            loss_test1 = criterion(outputtest1.cpu(),y_test1)
            acc_test1 = accuracy_Nclass(outputtest1.cpu().detach().numpy(),y_test1.detach().numpy())
            
            outputtest2 = NN_EWC1(x_test2)
            loss_test2 = criterion(outputtest2.cpu(),y_test2)
            acc_test2 = accuracy_Nclass(outputtest2.cpu().detach().numpy(),y_test2.detach().numpy())
            
            print(f'epoch {epoch}/{num_epochs}, step {i+1}/{total_steps}, accuracy test1 {acc_test1} %, lost test1, {loss_test1.item()}')
            print(f'accuracy test2 {acc_test2} %, lost test2, {loss_test2.item()}')
            
                # Store the results
            RES1_EWC += [[epoch+50, loss_test1.item(),acc_test1,loss.item(), acc]] 
            RES2_EWC += [[epoch+50, loss_test2.item(),acc_test2]] 

RES1_EWC = np.vstack(RES1_EWC)
RES2_EWC = np.vstack(RES2_EWC)

plt.figure(figsize=[8,5]) # Increase the size of the plots
plt.rcParams.update({'font.size': 18})
plt.plot(RES1[:,0],RES1[:,2],label="Task A")
plt.plot(RES2[:,0],RES2[:,2],label="Task B")
plt.plot(RES1_EWC[:,0],RES1_EWC[:,2],label="Task A EWC") # $\lambda$ = 100")
plt.plot(RES2_EWC[:,0],RES2_EWC[:,2],label="Task B EWC") # $\lambda$ = 100")
plt.ylim([80,100])
plt.xlim([0,100])
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("EWC.pdf")

