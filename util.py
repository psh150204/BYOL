import math
import torch
import numpy as np

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def warmup_learning_rate(opt, epoch, batch_id, total_batches, optimizer):
    if opt.warm and epoch <= opt.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (opt.warm_epochs * total_batches)
        lr = opt.warmup_from + p * (opt.warmup_to - opt.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, opt):
    """Decay the learning rate based on schedule"""
    lr = opt.lr
    if opt.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch-opt.warm_epochs) / (opt.epochs-opt.warm_epochs)))
    else:  # multi-step lr schedule
        steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
        if steps > 0:
            lr = lr * (opt.lr_decay_rate ** steps)
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }

    torch.save(state, save_file)
    print('Checkpoint saved to ' + save_file)
    del state