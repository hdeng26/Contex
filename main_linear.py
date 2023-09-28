from __future__ import print_function

import random
import numpy as np
import sys
import argparse
import time
import math
import os
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets, models
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy, accuracy_per_class
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def percentage_label(percent, label):
    set_seed()
    num_remove = int((1-percent/100)*label.size(dim=0))
    replace_uniq = torch.arange(label.max()+1, label.max()+num_remove+1).cuda()
    g_cuda = torch.Generator(device='cuda')
    indices = torch.randperm(label.size(dim=0), generator=g_cuda, device='cuda')[:num_remove]

    # find preserve indices
    batch_idx = torch.arange(label.size(dim=0)).cuda()
    mask = torch.ones(batch_idx.numel(), dtype=torch.bool).cuda()
    mask[indices] = False

    '''tensor([   1,   34,   40,   42,   47,   68,   75,   95,   97,  101,  113,  116,
         138,  139,  141,  148,  169,  201,  206,  217,  224,  249,  259,  281,
         284,  332,  341,  344,  347,  361,  371,  400,  402,  408,  420,  433,
         434,  444,  458,  466,  470,  478,  481,  497,  501,  503,  513,  517,
         520,  522,  528,  549,  550,  558,  581,  587,  594,  611,  616,  633,
         645,  646,  653,  675,  679,  706,  719,  720,  744,  749,  756,  760,
         773,  784,  790,  793,  809,  818,  822,  850,  860,  871,  875,  883,
         902,  903,  913,  914,  919,  920,  922,  952,  959,  968,  970,  972,
         983,  986,  988, 1013, 1020, 1032, 1048, 1056, 1069, 1076, 1079, 1086,
        1098, 1100, 1120, 1133, 1137, 1152, 1171, 1179, 1194, 1195, 1203, 1205,
        1213, 1221, 1244, 1255, 1259, 1264, 1269, 1271, 1291, 1298, 1314, 1317,
        1320, 1323, 1326, 1328, 1331, 1355, 1362, 1392, 1403, 1427, 1431, 1432],
       device='cuda:0')'''
    return batch_idx[mask], label.index_add(0, indices, replace_uniq)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=14,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'sun', 'food', 'DTD', 'caltech101', 'caltech256',  'path'], help='dataset')
    parser.add_argument('--data_folder', type=str, default='./datasets/', help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    # other setting
    parser.add_argument('--percent', type=int, default=100,
                        help='percentage of labels')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--opt_type', type=str, default='sgd',
                        help='optim to use')

    parser.add_argument('--acc_per_class', action='store_true',
                        help='use mean per class accuracy')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--finetune', action='store_true',
                        help='liner or whole network')
    parser.add_argument('--keep_size', action='store_true',
                        help='maintain the input image size')
    opt = parser.parse_args()
    
    #if opt.finetune:
    #    opt.learning_rate = 0.05*opt.batch_size/256
    #    opt.lr_decay_epochs = '1000'
    #else:
    #    opt.learning_rate = 0.1 * opt.batch_size / 256


    
    tb_dir = "tensorboard_" + opt.data_folder + opt.dataset
    opt.tb_folder = os.path.join(opt.ckpt, tb_dir)
    opt.ckpt = opt.ckpt + "last.pth"
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.finetune:
        opt.model_name = '{}_finetune'.format(opt.model_name)
        
    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'sun':
        opt.n_cls = 397
    elif opt.dataset == 'DTD':
        opt.n_cls = 47
    elif opt.dataset == 'food':
        opt.n_cls = 101
    elif opt.dataset == 'caltech101':
        opt.n_cls = 101
    elif opt.dataset == 'caltech256':
        opt.n_cls = 256
    elif opt.dataset == 'path':
        opt.n_cls = 1000
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

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
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.size == 32 and opt.finetune and not opt.keep_size:
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize, ])

        val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize, ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    val_transform_imagenet = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(opt.size),
        transforms.ToTensor(),
        normalize,
    ])

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
    elif opt.dataset == 'sun':
        train_dataset = datasets.SUN397(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.StanfordCars(root=opt.data_folder,
                                        transform=val_transform)
    elif opt.dataset == 'food':
        train_dataset = datasets.Food101(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.Food101(root=opt.data_folder,
                                        transform=val_transform)
    elif opt.dataset == 'DTD':
        train_dataset = datasets.DTD(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.Food101(root=opt.data_folder,
                                        transform=val_transform)
    elif opt.dataset == 'caltech101':
        train_dataset = datasets.Caltech101(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.Caltech101(root=opt.data_folder,
                                        transform=val_transform)
    elif opt.dataset == 'caltech256':
        train_dataset = datasets.Caltech256(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.Caltech256(root=opt.data_folder,
                                        transform=val_transform)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder+"Training_set",
                                            transform=train_transform)
        val_dataset = datasets.ImageFolder(root=opt.data_folder+"Validation_set",
                                             transform=val_transform_imagenet)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    set_seed()
    indices = torch.randperm(len(train_dataset))
    train_random_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_loader = torch.utils.data.DataLoader(
        train_random_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader

class cifar10Resnet50(torch.nn.Module):
    def __init__(self, opt):
        super(cifar10Resnet50, self).__init__()
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.encoder = SupConResNet(name=opt.model).encoder
        self.head = SupConResNet(name=opt.model).head
    def forward(self, x):
        x = self.upsample(x)
        x = self.encoder(x)
        x = torch.nn.functional.normalize(self.head(torch.squeeze(x)), dim=1)
        return x


def set_model(opt):
    model = SupConResNet(name=opt.model)

    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    # adjust for cifar
    if opt.size == 32 and opt.model == 'resnet50':
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        model.load_state_dict(state_dict)

        #debug weights
        # res50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # model.encoder = torch.nn.Sequential(*list(res50.children())[:-1])

        model.encoder[0] = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        del model.encoder[3]
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        cudnn.benchmark = True
        model = model.cuda()

    elif torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        model.load_state_dict(state_dict)


    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    if opt.finetune == True:
        model.train()
    else:
        model.eval()
    classifier.train()

    model.encoder.requires_grad = True

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # remove no label data
        remain_indices, _ = percentage_label(opt.percent, labels)
        images = torch.index_select(images, 0, remain_indices)
        labels = torch.index_select(labels, 0, remain_indices)

        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        if opt.finetune:
            features = model.encoder(images)
            output = classifier(torch.squeeze(features))

        else:
            with torch.no_grad():
                features = model.encoder(images)
            if opt.model == "resnet50t4" or opt.model == "resnet101t4":
                output = classifier(features.detach())
            else:
                output = classifier(torch.squeeze(features.detach()))
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()
    best_top1 = 0
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        if opt.acc_per_class:
            correct_all = torch.zeros(opt.n_cls).cuda()
            total_all = torch.zeros(opt.n_cls).cuda()
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            if opt.model == "resnet50t4" or opt.model == "resnet101t4":
                output = classifier(model.encoder(images))
            else:
                output = classifier(torch.squeeze(model.encoder(images)))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            if opt.acc_per_class:
                correct, total = accuracy_per_class(output, labels, opt.n_cls ,topk=(1,))
                correct_all += correct
                total_all += total
                topOne = torch.mean(correct[total != 0]/total[total != 0])*100
                top1.update(topOne, bsz)
            else:
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top1.update(acc1[0], bsz)
                top5.update(acc5[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    if opt.acc_per_class:
        top1_all = torch.mean(correct_all[total_all != 0] / total_all[total_all != 0])*100
        print(' * Acc@1 {top1:.3f}'.format(top1=top1_all))

    return losses.avg, top1.avg, top5.avg


def main():
    best_acc = 0
    opt = parse_option()
    set_seed()
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)


    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()

        # tensorboard logger
        logger.log_value('training loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        logger.log_value('training accuracy', acc, epoch)

        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        loss, val_top1, val_top5 = validate(val_loader, model, classifier, criterion, opt)
        logger.log_value('testing loss', loss, epoch)
        logger.log_value('testing top 1 accuracy', val_top1, epoch)
        logger.log_value('testing top 5 accuracy', val_top5, epoch)
        if val_top1 > best_acc:
            best_acc = val_top1

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
