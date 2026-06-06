import os
import time
import torch
import logging
import argparse
import warnings
import numpy as np
import torchattacks
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score
warnings.filterwarnings("ignore", category=FutureWarning)

def evaluate_model(model, dataset_loader, attack=None, attack_name="Clean", device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    y_true_all, y_pred_all = [], []

    for i, (x, y) in enumerate(dataset_loader):
        x, y = x.to(device), y.to(device)

        if attack is not None:
            x = attack(x, y)

        with torch.no_grad():
            outputs = model(x)
            predicted_class = outputs.argmax(dim=1)

        y_true_all.extend(y.cpu().numpy())
        y_pred_all.extend(predicted_class.cpu().numpy())

    acc = (torch.tensor(y_true_all) == torch.tensor(y_pred_all)).float().mean().item()
    precision = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)

    print(f"\n=== {attack_name} RESULTS ===")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%\n")

    return acc, precision, recall, f1

def accuracy_clean(model, dataset_loader, device):
    return evaluate_model(model, dataset_loader, None, "Clean", device)

def accuracy_FGSM(model, dataset_loader, eps, device, normalize):
    attack = torchattacks.FGSM(model, eps=eps)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if normalize:
        attack.set_normalization_used(mean=mean, std=std)

    return evaluate_model(model, dataset_loader, attack, f"FGSM (ε={eps})", device)

def accuracy_PGD(model, dataset_loader, eps, device, normalize):
    attack = torchattacks.PGD(model, eps=eps)   
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if normalize:
        attack.set_normalization_used(mean=mean, std=std)

    return evaluate_model(model, dataset_loader, attack, f"PGD (ε={eps})", device)

def accuracy_MIM(model, dataset_loader, eps, device, normalize):
    attack = torchattacks.MIFGSM(model, eps=eps) 
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if normalize:
        attack.set_normalization_used(mean=mean, std=std)

    return evaluate_model(model, dataset_loader, attack, f"MIM (ε={eps})", device)
    
def accuracy_AutoAttack(model, dataset_loader, num_classes, eps, device, normalize):
    attack = torchattacks.AutoAttack(model, eps=eps, n_classes=num_classes)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if normalize:
        attack.set_normalization_used(mean=mean, std=std)

    return evaluate_model(model, dataset_loader, attack, f"AutoAttack (ε={eps})", device)

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=50)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--env', type=str, choices=['colab', 'kaggle'], default='kaggle')
parser.add_argument('--dataset', default='lisa', type=str)
parser.add_argument('--normalize', action='store_true', help='Ativa normalização dos dados')
parser.add_argument('-ia', '--ignore_autoattack', action='store_true')
parser.add_argument('--save', type=str, default='./experiment')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('-tl', '--total_loops', type=int, default=1)
args = parser.parse_args()

print(args)

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut

class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

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

def compute_gradient_penalty(model, inputs):
    inputs = inputs.clone().detach().requires_grad_(True)
    
    outputs = model(inputs)
    if outputs.ndim == 2:  # Caso saída seja (B, C)
        outputs = outputs.norm(2, dim=1).mean()
    else:
        outputs = outputs.mean()

    gradients = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty

    
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        
def lisa_loaders(batch_size=32, normalize=False):
    train_dir = "/kaggle/input/cropped-lisa-traffic-light-dataset/cropped_lisa_1/train_1"
    val_dir = "/kaggle/input/cropped-lisa-traffic-light-dataset/cropped_lisa_1/val_1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_list = [transforms.Resize((64, 32)), transforms.ToTensor()]

    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )

    transform = transforms.Compose(transform_list)
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    train_eval_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False, num_workers=2)

    return train_loader, test_loader, train_eval_loader, 7
    
def bstl_loaders(env, batch_size=128, normalize=False):
    if env == "colab":
        train_dir = "/content/archive/train"
        test_dir = "/content/archive/test"
    else:
        train_dir = "/kaggle/input/bstl-dataset/train"
        test_dir = "/kaggle/input/bstl-dataset/test"
    
    transform_list = [transforms.Resize((64, 32)), transforms.ToTensor()]

    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )

    transform = transforms.Compose(transform_list)

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)
    train_eval_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False, num_workers=2)

    print(f"Número de imagens em train: {len(train_dataset)}")
    print(f"Número de imagens em test: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    return train_loader, test_loader, train_eval_loader, 4

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def accuracy(model, dataset_loader, device):
    total_correct = 0
    total_samples = 0
    for images, labels in dataset_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    return total_correct / total_samples

if __name__ == '__main__':

    makedirs(args.save)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)   
    
    is_odenet = args.network == 'odenet'
    if args.dataset == "lisa":
        train_loader, test_loader, train_eval_loader, num_classes = lisa_loaders(512, args.normalize) 
    else:
        train_loader, test_loader, train_eval_loader, num_classes = bstl_loaders(args.env, 1024, args.normalize) 

    if args.downsampling_method == 'conv':
        downsampling_layers = [
            nn.Conv2d(3, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        ]
    elif args.downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(3, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]

    for loop_idx in range(args.total_loops):
        if args.total_loops > 1:
            print(f"\n========== Loop {loop_idx + 1}/{args.total_loops} ==========\n")
            
        feature_layers = [ODEBlock(ODEfunc(64))] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
        fc_layers = [
            norm(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(64, num_classes)
        ]
    
        model = nn.Sequential(
            *downsampling_layers,
            *feature_layers,
            *fc_layers
        ).to(device)
    
        if torch.cuda.device_count() > 1:
            print(f"Usando {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
    
        criterion = nn.CrossEntropyLoss().to(device)
    
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9
        )
    
        best_acc = 0
    
        data_gen = inf_generator(train_loader)
        batches_per_epoch = len(train_loader)
    
        lr_fn = learning_rate_with_decay(
            args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
            decay_rates=[1, 0.1, 0.01, 0.001]
        )
    
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        best_acc = 0
        batch_time_meter = RunningAverageMeter()
        f_nfe_meter = RunningAverageMeter()
        b_nfe_meter = RunningAverageMeter()
        end = time.time()
        
        for itr in range(args.nepochs * batches_per_epoch):
    
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_fn(itr)
    
            optimizer.zero_grad()
            x, y = data_gen.__next__()
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
    
            if is_odenet:
                nfe_forward = feature_layers[0].nfe
                feature_layers[0].nfe = 0
    
            loss.backward()
            optimizer.step()
    
            if is_odenet:
                nfe_backward = feature_layers[0].nfe
                feature_layers[0].nfe = 0
    
            batch_time_meter.update(time.time() - end)
            if is_odenet:
                f_nfe_meter.update(nfe_forward)
                b_nfe_meter.update(nfe_backward)
            end = time.time()
    
            if itr % batches_per_epoch == 0:
                with torch.no_grad():
    
                    train_acc = accuracy(model, train_eval_loader, device)
                    val_acc = accuracy(model, test_loader, device)
    
                    if val_acc > best_acc:
                        torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, f'ode_{args.dataset}_{loop_idx + 1}.pth'))
    
                        best_acc = val_acc
            
                    print(
                        "Epoch {} | Time {:.3f} ({:.3f}) | Train Acc {:.2f}% | Test Acc {:.2f}%".format(
                            itr // batches_per_epoch + 1,
                            batch_time_meter.val,
                            batch_time_meter.avg,
                            train_acc * 100,
                            val_acc * 100
                        )
                    )

    all_eps = [0.01, 8/255, 0.04, 0.055, 0.07, 0.085, 0.1, 0.115, 0.13, 0.15, 0.175, 0.2]
    for loop_idx in range(args.total_loops):
        if args.total_loops > 1:
            print(f"\n{'='*50}")
            print(f" Avaliando modelo {loop_idx + 1}/{args.total_loops}")
            print(f"{'='*50}\n")
            
        ckpt_path = os.path.join(args.save, f"ode_{args.dataset}_{loop_idx + 1}.pth")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.eval()
    
        evaluate_model(model, test_loader, device=device)
    
        for eps in all_eps:
            acc, prec, rec, f1 = accuracy_FGSM(model, test_loader, eps, device, args.normalize)
    
        for eps in all_eps:
            acc, prec, rec, f1 = accuracy_PGD(model, test_loader, eps, device, args.normalize)
    
        for eps in all_eps:
            acc, prec, rec, f1 = accuracy_MIM(model, test_loader, eps, device, args.normalize)
            
        if args.ignore_autoattack:
            for eps in all_eps:
                acc, prec, rec, f1 = accuracy_AutoAttack(model, test_loader, num_classes, eps, device, args.normalize)
