import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np

train_dataset_path='../input/koronannamk2/training'
test_dataset_path='../input/koronannamk2/test'
mean = [0.485, 0.456, 0.406]
std =  [0.229, 0.224, 0.225]
train_transforms=transforms.Compose([
        transforms.Resize((255)),
transforms.CenterCrop(224) ,
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
])
test_transforms=transforms.Compose([
          transforms.Resize((255)),
         transforms.ToTensor(),
         transforms.Normalize(torch.Tensor(mean),
                              torch.Tensor(std)
        ) ])
train_dataset=torchvision.datasets.ImageFolder(root=train_dataset_path,transform=train_transforms)
test_dataset=torchvision.datasets.ImageFolder(root=test_dataset_path,transform=test_transforms)

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=False)


def show(dataset):
    loader=torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=True)
    batch=next(iter(loader))
    images,labels=batch
    grid=torchvision.utils.make_grid(images,nrow=3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid,(1,2,0)))
    print(labels)
show(train_dataset)


def set_device():
    if torch.cuda.is_available():
        dev="cuda:0"
    else:
        dev="cpu"
    return torch.device(dev)

def train_nn(model,train_loader,test_loader,loss_fn,optimizer,num_epochs):
    model.train()
    device=set_device()
    for epoch in range(num_epochs):
        running_loss=0.0
        running_correct=0.0
        total=0
        for data in train_loader:
            images,labels=data
            images=images.to(device)
            labels=labels.to(device)
            total+=labels.size(0)
            optimizer.zero_grad()
            outputs=model(images)
            _,predicted=torch.max(outputs.data,1)
            loss=loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
           
            running_correct+=(labels==predicted).sum().item()
            
        epoch_loss=running_loss/len(train_loader)

        epoch_acc=100*running_correct/total
        print("   - Training dataset.Got %d out of %d images correctly (%.3f%%).Epoch loss: %.3f"%(running_correct,total,epoch_acc,epoch_loss))
        evuluate_model(model,test_loader)
    return model

def evuluate_model(model,test_loader):
    model.eval()
    predicted_correct=0
    total=0
    device=set_device()
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            images=images.to(device)
            labels=labels.to(device)
            total+=labels.size(0)
            outputs=model(images)
            _,predicted=torch.max(outputs.data,1)
            predicted_correct+=(labels==predicted).sum().item()
            epoch_acc=100*predicted_correct/total
            


resnet18_model=models.resnet18(pretrained=False)
number_features=resnet18_model.fc.in_features
number_of_classes=2
resnet18_model.fc=nn.Linear(number_features,number_of_classes)
device=set_device()
resnet18_model=resnet18_model.to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=optim.SGD(resnet18_model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.03)
train_nn(resnet18_model,train_loader,test_loader,loss_fn,optimizer,20)
