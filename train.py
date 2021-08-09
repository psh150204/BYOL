import argparse, os, math, time, sys

import torch
import torch.nn as nn

from util import TwoCropTransform, save_model, AverageMeter
from util import warmup_learning_rate, adjust_learning_rate
from models import model_dict

from torchvision import transforms, datasets

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimizer
    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD')

    # model / dataset
    parser.add_argument('--model', type=str, default='ResNet50')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--img_size', type=int, default=32,
                        help='parameter for RandomResizedCrop')
    
    # others
    parser.add_argument('--trial', type=int, default=0,
                        help='trial number')
    
    opt = parser.parse_args()

    # lr scheduling
    opt.warm = True
    opt.cos = True
    opt.warmup_to = opt.lr
    opt.warmup_from = 0

    # set the path according to the environment
    opt.model_path = './save/{}_models'.format(opt.dataset)

    opt.model_name = 'BYOL_{}_lr{}_wd{}_bsz{}_ep{}_cos_warm_trial{}'.\
        format(opt.model, opt.lr, opt.wd, opt.batch_size, opt.epochs, opt.trial)
    
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    opt.data_folder = './data'
    
    # device setting
    opt.device = torch.device('cpu')
    if torch.cuda.is_available():
        opt.device = torch.device('cuda')

    # number of classes
    if opt.dataset == 'cifar10':
        opt.num_classes = 10
    elif opt.dataset == 'cifar100':
        opt.num_classes = 100

    return opt

def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    normalize = transforms.Normalize(mean=mean, std=std)

    # SimCLR style augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    else:
        raise ValueError(opt.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader

def set_model(opt):
    # TODO : set up your model
    # TIP : you can load the ResNet-like models using "model = model_dict[opt.model](num_classes=opt.num_classes)"
    
    raise NotImplementedError

    return model

def set_optimizer(opt, model, num_iter):
    optimizer = torch.optim.SGD(model.parameters(),
                 lr=opt.lr, momentum=opt.momentum, weight_decay=opt.wd)

    return optimizer

def train(train_loader, model, optimizer, epoch, opt, writer):
    """ one epoch training """

    model.train()
    for idx, ((x1, x2), _) in enumerate(train_loader):
        x1, x2 = x1.to(opt.device), x2.to(opt.device)

        # warmup step
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # TODO : training the model

        raise NotImplementedError

def main():
    opt = parse_option()
    train_loader = set_loader(opt)
    model = set_model(opt)
    optimizer= set_optimizer(opt, model, len(train_loader))

    for epoch in range(1, opt.epochs+1):
        end = time.time()

        # cosine scheduling
        adjust_learning_rate(optimizer, epoch, opt)

        train(train_loader, model, optimizer, epoch, opt, writer)
        print('epoch {}, total time {:.2f}s'.format(epoch, time.time()-end))

        if epoch % 100 == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

if __name__ == '__main__':
    main()
