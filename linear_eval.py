import sys, argparse, time, math

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from util import AverageMeter, accuracy, adjust_learning_rate
from models import model_dict
from PIL import Image

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--lr', type=float, default=30.,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='ResNet50')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'], help='dataset')

    # other setting
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './data/'

    opt.model_name = '{}_{}_eval_from_{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.ckpt, opt.type, opt.lr, opt.weight_decay,
               opt.batch_size)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
    opt.lr_decay_epochs = '80,90,95'
    opt.lr_decay_rate = 0.2
    opt.cos = False

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

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
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader

def set_model(opt):
    model = model_dict[opt.model](num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()

    # TODO : Load the pretrained model
    print('==> loading pre-trained model')

    raise NotImplementedError
    
    print('==> done')

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    
    return model, criterion

def set_optimizer(opt, model):
    opt.lr = 30.0 * opt.batch_size / 256

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['linear.weight', 'linear.bias']:
            param.requires_grad = False

    # init the fc layer
    model.linear.weight.data.normal_(mean=0.0, std=0.01)
    model.linear.bias.data.zero_()

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # linear.weight, linear.bias

    optimizer = torch.optim.SGD(parameters, lr=opt.lr,
                                momentum=opt.momentum, weight_decay=opt.weight_decay)
    
    return optimizer

def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        # TODO : training the linear classifier
        raise NotImplementedError

def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            
            # forward
            with torch.no_grad():
                output = model(images).detach()
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]  '
                      'Time {time:.3f}  '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), time=time.time()-end,
                       loss=losses, top1=top1))
                
                end = time.time()

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    opt = parse_option()
    
    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(optimizer, epoch, opt)
        
        # train for one epoch
        time1 = time.time()
        train(train_loader, model, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}'.format(
            epoch, time2 - time1))

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc

        print('best accuracy: {:.2f}'.format(best_acc))
    print('Linear Evaluation Finished with type {} : evaluated from {}'.format(opt.type, opt.ckpt))
    
if __name__ == '__main__':
    main()