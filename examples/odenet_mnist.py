import os
import math
import random
import argparse
import logging
import numpy as np
import torch
import geotorch
import kagglehub
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import *
from datasets import *
from sodef_utils import *
from types import SimpleNamespace
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint_adjoint as odeint
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import MNIST, CIFAR10, ImageFolder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
fc_dim = 64
args = get_args() 
seed_torch()

device = 'cuda' 
best_acc = 0 

if args.dataset == 'lisa':
    trainloader, testloader, train_eval_loader, testset, num_classes = lisa_loaders(train_batch_size=args.batch_size, test_batch_size=args.batch_size, normalize=args.normalize)
elif args.dataset == 'cifar10':
    trainloader, testloader, train_eval_loader, testset, num_classes = cifar10_loaders(train_batch_size=args.batch_size, test_batch_size=args.batch_size, normalize=args.normalize )
else:
    trainloader, testloader, train_eval_loader, testset, num_classes = bstl_loaders(train_batch_size=args.batch_size, test_batch_size=1024, normalize=args.normalize)

print('==> Building model..')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for loop_idx in range(args.total_loops):
    if args.total_loops > 1:
        print(f"\n========== Loop {loop_idx + 1}/{args.total_loops} ==========\n")
            
    net = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    net = nn.Sequential(*list(net.children())[0:-1])
    
    fcs_temp = fcs()
    fc_layers = MLP_OUT_BALL(num_classes)
        
    net = nn.Sequential(*net, fcs_temp, fc_layers).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, eps=1e-4, amsgrad=True)
    
    def save_feature(model, dataset_loader, save_path):
        x_save, y_save = [], []
        modulelist = list(model)
    
        for x, y in dataset_loader:
            x = x.to(device)
            y_ = y.numpy()
    
            for l in modulelist[:-2]:
                x = l(x)
    
            x = net[-2](x[..., 0, 0])
            x_ = x.cpu().detach().numpy()
    
            x_save.append(x_)
            y_save.append(y_)
    
        np.savez(save_path, x_save=np.concatenate(x_save), y_save=np.concatenate(y_save))
    
    def train(epoch, trainloader):
        net.train()
        train_loss, correct, total = 0, 0, 0
        modulelist = list(net)
    
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
    
            x = inputs
            for l in modulelist[:-2]:
                x = l(x)
    
            x = net[-2](x[..., 0, 0])
            x = net[-1](x)
            outputs = x
    
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
        acc = 100. * correct / total
        print(f"\nTrain in epoch {epoch+1}: accuracy = {round(acc, 2)}%")
    
    def test(epoch, testloader, save_model_path, train_eval_loader):
        global best_acc
        net.eval()
        test_loss, correct, total = 0, 0, 0
        modulelist = list(net)
    
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
    
                x = inputs
                for l in modulelist[:-2]:
                    x = l(x)
    
                x = net[-2](x[..., 0, 0])
                x = net[-1](x)
                outputs = x
    
                loss = criterion(outputs, targets)
                test_loss += loss.item()
    
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
                
        acc = 100. * correct / total
        print(f"Test in epoch {epoch+1}: accuracy = {round(acc, 2)}%")
        if acc > best_acc:
            state = {'net': net.state_dict(), 'acc': acc, 'epoch': epoch,}
            torch.save(state, args.folder_savemodel + f'/extractor{args.total_loops}.pth')
            best_acc = acc
    
            save_feature(net, train_eval_loader, args.train_savepath)
            save_feature(net, testloader, args.test_savepath)
    
    ############################################### Phase 1 ################################################
    makedirs(args.folder_savemodel)
    makedirs('./data')
    
    #for epoch in range(args.epochs_phase1):
        #train(epoch, trainloader)
        #test(epoch, testloader, './models/ckpt.pth', train_eval_loader)
        
    ################################################ Phase 2 ################################################
    weight_diag = 10
    weight_offdiag = 10
    weight_norm = 0
    weight_lossc = 0
    weight_f = 0.2
    
    exponent = 1.0
    exponent_f = 50
    exponent_off = 0.1
    
    endtime = 1
    
    trans = 1.0
    transoffdig = 1.0
    trans_f = 0.0
    numm = 8
    timescale = 1
    fc_dim = 64
    t_dim = 1
    act = torch.sin
    act2 = torch.nn.functional.relu
    
    class ConcatFC(nn.Module):
    
        def __init__(self, dim_in, dim_out):
            super(ConcatFC, self).__init__()
            self._layer = nn.Linear(dim_in, dim_out)
    
        def forward(self, t, x):
            return self._layer(x)
    
    class ODEfunc_mlp(nn.Module): 
    
        def __init__(self, dim):
            super(ODEfunc_mlp, self).__init__()
            self.fc1 = ConcatFC(fc_dim, fc_dim)
            self.act1 = act
            self.nfe = 0
    
        def forward(self, t, x):
            self.nfe += 1
            out = -1 * self.fc1(t, x)
            out = self.act1(out)
            return out
    
    class ODEBlocktemp(nn.Module):
    
        def __init__(self, odefunc):
            super(ODEBlocktemp, self).__init__()
            self.odefunc = odefunc
            self.integration_time = torch.tensor([0, endtime]).float()
    
        def forward(self, x):
            out = self.odefunc(0, x)
            return out
    
        @property
        def nfe(self):
            return self.odefunc.nfe
    
        @nfe.setter
        def nfe(self, value):
            self.odefunc.nfe = value
    
    class Flatten(nn.Module):
    
        def __init__(self):
            super(Flatten, self).__init__()
    
        def forward(self, x):
            shape = torch.prod(torch.tensor(x.shape[1:])).item()
            return x.view(-1, shape)
    
    class MLP_OUT(nn.Module):
    
        def __init__(self, num_classes=10):
            super(MLP_OUT, self).__init__()
            self.fc0 = nn.Linear(fc_dim, num_classes)
    
        def forward(self, input_):
            h1 = self.fc0(input_)
            return h1
    
    def one_hot(x, K):
        return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)
    
    def accuracy(model, dataset_loader, num_classes):
        total_correct = 0
        for x, y in dataset_loader:
            x = x.to(device)
            y = one_hot(np.array(y.numpy()), num_classes)
    
            target_class = np.argmax(y, axis=1)
            predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
            total_correct += np.sum(predicted_class == target_class)
        return total_correct / len(dataset_loader.dataset)
    
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def df_dz_regularizer(f, z):
        regu_diag = 0.
        regu_offdiag = 0.0
        for ii in np.random.choice(z.shape[0], min(numm, z.shape[0]), replace=False):
            batchijacobian = torch.autograd.functional.jacobian(lambda x: odefunc(torch.tensor(1.0).to(device), x),
                                                                z[ii:ii + 1, ...], create_graph=True)
            batchijacobian = batchijacobian.view(z.shape[1], -1)
            if batchijacobian.shape[0] != batchijacobian.shape[1]:
                raise Exception("wrong dim in jacobian")
    
            tempdiag = torch.diagonal(batchijacobian, 0)
            regu_diag += torch.exp(exponent * (tempdiag + trans))
    
            offdiat = torch.sum(
                torch.abs(batchijacobian) * ((-1 * torch.eye(batchijacobian.shape[0]).to(device) + 0.5) * 2), dim=0)
            off_diagtemp = torch.exp(exponent_off * (offdiat + transoffdig))
            regu_offdiag += off_diagtemp
    
        return regu_diag / numm, regu_offdiag / numm
    
    
    def f_regularizer(f, z):
        tempf = torch.abs(odefunc(torch.tensor(1.0).to(device), z))
        regu_f = torch.pow(exponent_f * tempf, 2)
        return regu_f
    
    
    def critialpoint_regularizer(y1):
        regu4 = torch.linalg.norm(y1, dim=1)
        regu4 = regu4.mean()
        regu4 = torch.exp(-0.1 * regu4 + 5)
        return regu4.mean()
    
    class DenseDataset(Dataset):
        def __init__(self, savepath):
            npzfile = np.load(savepath)
    
            self.x = npzfile['x_save']
            self.y = npzfile['y_save']
    
        def __len__(self):
            return len(self.x)
    
        def __getitem__(self, idx):
            return self.x[idx, ...], self.y[idx]
    
    odesavefolder = './EXP/dense_resnet_final'
    makedirs(odesavefolder)
    odefunc = ODEfunc_mlp(0)
    
    feature_layers = [ODEBlocktemp(odefunc)]
    fc_layers = [MLP_OUT(num_classes)]
    
    for param in fc_layers[0].parameters():
        param.requires_grad = False
    
    model = nn.Sequential(*feature_layers, *fc_layers).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    regularizer = nn.MSELoss()
    
    train_loader = DataLoader(DenseDataset(args.train_savepath), batch_size=args.batch_size, shuffle=True, num_workers=1)
    train_loader__ = DataLoader(DenseDataset(args.train_savepath), batch_size=args.batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(DenseDataset(args.test_savepath), batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)
    
    best_acc = 0
    last_model_path = None

    for itr in range(args.epochs_phase2 * batches_per_epoch):
        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        modulelist = list(model)
        y0 = x
        x = modulelist[0](x)
        y1 = x
        for l in modulelist[1:]:
            x = l(x)
        logits = x
        y00 = y0  
    
        regu1, regu2 = df_dz_regularizer(odefunc, y00)
        regu1 = regu1.mean()
        regu2 = regu2.mean()
        regu3 = f_regularizer(odefunc, y00)
        regu3 = regu3.mean()
        loss = weight_f * regu3 + weight_diag * regu1 + weight_offdiag * regu2
    
        if itr % 100 == 1:
            torch.save({'state_dict': model.state_dict(), 'args': args},
                       os.path.join(odesavefolder, 'model_diag.pth' + str(itr // 100)))
    
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        if itr % batches_per_epoch == 0:
            print(f"Epoch {itr}/{args.epochs_phase2 * batches_per_epoch}")
        
            if itr == 0:
                continue
        
            with torch.no_grad():
                last_model_path = os.path.join(odesavefolder, f"model_{itr // batches_per_epoch}.pth")
        
                torch.save({'state_dict': model.state_dict(), 'args': args}, last_model_path)
    
    ################################################ Phase 3, train final FC ################################################
    endtime = 5
    layernum = 0
    
    folder = last_model_path
    saved = torch.load(folder, weights_only=False)
    print('load...', folder)
    statedic = saved['state_dict']
    args = saved['args']
    tol = 1e-5
    savefolder_fc = './EXP/resnetfct5_15/'
    print('saving...', savefolder_fc, ' endtime... ',endtime)
    
    class ODEBlock(nn.Module):
    
        def __init__(self, odefunc):
            super(ODEBlock, self).__init__()
            self.odefunc = odefunc
            self.integration_time = torch.tensor([0, endtime]).float()
    
        def forward(self, x):
            self.integration_time = self.integration_time.type_as(x)
            out = odeint(self.odefunc, x, self.integration_time, rtol=tol, atol=tol)
            return out[1]
    
        @property
        def nfe(self):
            return self.odefunc.nfe
    
        @nfe.setter
        def nfe(self, value):
            self.odefunc.nfe = value
    
    makedirs(savefolder_fc)
    odefunc = ODEfunc_mlp(0)
    feature_layers = [ODEBlock(odefunc)]
    fc_layers = [MLP_OUT(num_classes)]
    model = nn.Sequential(*feature_layers, *fc_layers).to(device)
    model.load_state_dict(statedic)
    for param in odefunc.parameters():
        param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss().to(device)
    regularizer = nn.MSELoss()
    
    train_loader = DataLoader(DenseDataset(args.train_savepath), batch_size=args.batch_size, shuffle=True, num_workers=1)
    train_loader__ = DataLoader(DenseDataset(args.train_savepath), batch_size=args.batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(DenseDataset(args.test_savepath), batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)
    
    best_acc = 0
    for itr in range(args.epochs_phase3 * batches_per_epoch):
    
        optimizer.zero_grad()
        x, y = data_gen.__next__()
        
        x = x.to(device)
        y = y.to(device)
    
        modulelist = list(model)
        
        y0 = x
        x = modulelist[0](x)
        y1 = x
        for l in modulelist[1:]:
            x = l(x)
        logits = x
    
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    
        if itr % batches_per_epoch == 0:
            if itr == 0:
                continue
            with torch.no_grad():
                val_acc = accuracy(model, test_loader, num_classes)
                train_acc = accuracy(model, train_loader__, num_classes)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(savefolder_fc, f'sodef_dense{args.total_loops}.pth'))
                    best_acc = val_acc
                print("Epoch {:04d}|Train Acc {:.4f} | Test Acc {:.4f}".format(itr // batches_per_epoch, train_acc, val_acc))
