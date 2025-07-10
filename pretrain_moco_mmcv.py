import argparse
import math
import os
import random
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import torch.distributed as dist


import moco.mmcv
from torch.utils.tensorboard import SummaryWriter
from dataset import get_pretraining_set_intra_cmq

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[100, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Distributed
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--checkpoint-path', default='./checkpoints', type=str)
parser.add_argument('--skeleton-representation', type=str,
                    help='input skeleton-representation  for self supervised training (image-based or graph-based or seq-based)')
parser.add_argument('--pre-dataset', default='ntu60', type=str,
                    help='which dataset to use for self supervised training (ntu60 or ntu120)')
parser.add_argument('--protocol', default='cross_subject', type=str,
                    help='traiining protocol cross_view/cross_subject/cross_setup')

# contrast specific configs:
parser.add_argument('--contrast-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--contrast-k', default=32768, type=int,
                    help='queue size; number of negative keys (default: 16384)')
parser.add_argument('--contrast-mk', default=0.999, type=float,
                    help='contrast momentum of updating k-encoder (default: 0.999)')
parser.add_argument('--contrast-mp', default=0.995, type=float,
                    help='contrast momentum of updating p-encoder (default: 0.995)')
parser.add_argument('--contrast-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--teacher-t', default=0.05, type=float,
                    help='softmax temperature (default: 0.05)')
parser.add_argument('--student-t', default=0.1, type=float,
                    help='softmax temperature (default: 0.1)')
parser.add_argument('--alpha', default=1.0, type=float,
                    help='weight of sim loss (default: 1.0)')
parser.add_argument('--beta', default=0.25, type=float,
                    help='weight of sim loss (default: 0.25)')
parser.add_argument('--topk', default=1024, type=int,
                    help='number of contrastive context')
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')



def init_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    cudnn.deterministic = True
    cudnn.benchmark = True

def main():
    args = parser.parse_args()

    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)


    # Simply call main_worker function
    main_worker(args)

def main_worker(args):
    if args.local_rank != -1:
        init_seeds(args.seed + args.local_rank)
    else:
        init_seeds(args.seed)

    # pretraining dataset and protocol
    from options import options_pretraining as options 
    if args.pre_dataset == 'ntu60' and args.protocol == 'cross_view':
        opts = options.opts_ntu_60_cross_view()
    elif args.pre_dataset == 'ntu60' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_60_cross_subject()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_setup':
        opts = options.opts_ntu_120_cross_setup()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_120_cross_subject()
    elif args.pre_dataset == 'pku_v2' and args.protocol == 'cross_view':
        opts = options.opts_pku_v2_cross_view()
    elif args.pre_dataset == 'pku_v2' and args.protocol == 'cross_subject':
        opts = options.opts_pku_v2_cross_subject()

    opts.train_feeder_args['input_representation'] = args.skeleton_representation

    # create model
    print("=> creating model")

    model = moco.mmcv.MoCo(args.skeleton_representation, opts.bi_gru_model_args,
                           args.contrast_dim, args.contrast_k, args.contrast_t,
                           args.teacher_t, args.student_t, args.alpha, args.beta, args.topk, args.contrast_mk,
                           args.contrast_mp, args.mlp)
    print("options",opts.train_feeder_args)

    print(model)

    model.cuda()
    if args.local_rank != -1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.distributed.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        print('Distributed data parallel model used')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ## Data loading code
    train_dataset = get_pretraining_set_intra_cmq(opts)

    if args.local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    def worker_init_fn(worker_id):
        return np.random.seed(torch.initial_seed()%(2**31) + worker_id)  # for single gpu
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers,
        worker_init_fn=worker_init_fn, pin_memory=True, sampler=train_sampler, drop_last=True)

    writer = SummaryWriter(args.checkpoint_path)

    for epoch in range(args.start_epoch, args.epochs):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss_info, loss_kd = train(train_loader, model, criterion, optimizer, epoch, args)
        
        if args.local_rank in [-1, 0]:
            writer.add_scalar('loss_info', loss_info.avg, global_step=epoch)
            writer.add_scalar('loss_kd', loss_kd.avg, global_step=epoch)

            if epoch % 10 == 0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best=False, filename=args.checkpoint_path+'/checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    losses_info = AverageMeter('Loss Info', ':6.3f')
    losses_kd = AverageMeter('Loss KD', ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses_info, losses_kd],
        prefix="Epoch: [{}] Lr_rate [{}]".format(epoch, optimizer.param_groups[0]['lr']))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_v1, input_v2, input_v3, input_v4) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs= [input_v1,input_v2, input_v3, input_v4]
        inputs[0] =inputs[0].float().cuda(non_blocking=True)
        inputs[1] =inputs[1].float().cuda(non_blocking=True)
        inputs[2] = inputs[2].float().cuda(non_blocking=True)
        inputs[3] = inputs[3].float().cuda(non_blocking=True)

        # compute output
        loss_info, loss_kd = model(inputs[0], inputs[1], inputs[2], inputs[3])

        batch_size = inputs[0].size(0)
        loss = loss_info + loss_kd

        losses.update(loss.item(), batch_size)
        losses_info.update(loss_info, batch_size)
        losses_kd.update(loss_kd.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses_info, losses_kd

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
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
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
