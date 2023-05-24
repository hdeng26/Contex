from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from util import adjust_learning_rate, warmup_learning_rate, accuracy, accuracy_per_class
from util import TwoCropTransform, AverageMeter, QuadCropTransform
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, LinearClassifier
from losses import SupConLoss0, SupConLoss1, SupConLoss2, SupConTupletLoss, SupConTupletLoss2, SupConTupletLoss3
from imagenet32Loader import ImageNetDownSample

#from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

#from torch.distributed.fsdp import (
#   FullyShardedDataParallel,
#   CPUOffload,
#)
#from torch.distributed.fsdp.wrap import (
#   size_based_auto_wrap_policy,
#)
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#try:
#    import apex
#    from apex import amp, optimizers
#except ImportError:
#    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=5,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='eval frequency')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # training efficiency
    parser.add_argument('--amp', action='store_true',
                        help='Automatic Mixed Precision')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--loss_type', type=int, default=0,
                        help='loss function to use')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'cars', 'food', 'imagenet32', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--load_ckpt', type=str, default=None, help='pretrained checkpoint')
    parser.add_argument('--resume', action='store_true', help='choose to load from ckpt or not')
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--acc_per_class', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()
    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'imagenet32':
        opt.n_cls = 1000
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
    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None
            #and opt.mean is not None \
            #and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = '../ckpts/SupCON/supplementary/{}_models'.format(opt.dataset)
    opt.tb_path = '../ckpts/SupCON/supplementary/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_loss_{}_decay_{}_bsz_{}_epoch{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate, opt.loss_type,
               opt.weight_decay, opt.batch_size, opt.epochs, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
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

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'imagenet32':
        # use imagenet std mean instead
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif opt.dataset == 'path':
        # use imagenet std mean instead
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        #mean = eval(opt.mean)
        #std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        #transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    linear_train_transform = transforms.Compose([
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
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
        linear_train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=linear_train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100': #TODO: change to quad transform and compare
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
        linear_train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                                transform=linear_train_transform,
                                                download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == 'imagenet32':
        train_dataset = ImageNetDownSample(root='/data/Dataset/ImageNet_32/Imagenet32_train',
                                          transform=TwoCropTransform(train_transform))
        linear_train_dataset = ImageNetDownSample(root='/data/Dataset/ImageNet_32/Imagenet32_train',
                                                transform=linear_train_transform)
        val_dataset = ImageNetDownSample(root='/data/Dataset/ImageNet_32/Imagenet32_val',
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == 'food':
        train_dataset = datasets.Food101(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
        linear_train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                                transform=linear_train_transform)
        val_dataset = datasets.ImageFolder(root=opt.data_folder + "Validation_set",
                                           transform=val_transform_imagenet)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    linear_train_loader = torch.utils.data.DataLoader(
        linear_train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)
    return train_loader, linear_train_loader, val_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    linear_criterion = torch.nn.CrossEntropyLoss()
    if opt.loss_type == 0:
        criterion = SupConLoss0(temperature=opt.temp)
    elif opt.loss_type == 1:
        criterion = SupConLoss1(temperature=opt.temp)
    elif opt.loss_type == -2:
        criterion = SupConTupletLoss2(temperature=opt.temp)
    elif opt.loss_type == -1:
        criterion = SupConTupletLoss(temperature=opt.temp)
    else:
        print("Error finding loss function\n")
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    # check for resume check point
    ckpt_path = os.path.join(opt.save_folder, 'best.pth')
    if opt.resume and os.path.exists(ckpt_path):
        # use resume ckpt instead
        opt.load_ckpt = ckpt_path

    if opt.load_ckpt is not None:
        ckpt = torch.load(opt.load_ckpt, map_location='cpu')
        state_dict = ckpt['model']
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model.encoder = torch.nn.DataParallel(model.encoder)
            else:
                new_state_dict = {}
                for k, v in state_dict.items():
                    k = k.replace("module.", "")
                    new_state_dict[k] = v
                state_dict = new_state_dict
            #if torch.cuda.device_count() > 1:
            #    model.encoder = torch.nn.DataParallel(model.encoder)
            model = model.cuda()
            criterion = criterion.cuda()
            classifier = classifier.cuda()
            cudnn.benchmark = True
            model.load_state_dict(state_dict)
    else:
        if torch.cuda.is_available():
            # DDP
            #dist.init_process_group("nccl")
            #rank = dist.get_rank()
            #device_id = rank % torch.cuda.device_count()
            #model = model.to(device_id)
            #criterion = criterion.to(device_id)
            #model.encoder = DDP(model.encoder, device_ids=[device_id])
            #model.encoder = torch.nn.parallel.DistributedDataParallel(model.encoder)
            # TODO: add pytorch 2.0 Torch.Compile
            # TODO: use fsdp to speed up the training
            #fsdp_model = FullyShardedDataParallel(
            #    SupConResNet(name=opt.model),
            #    cpu_offload=CPUOffload(offload_params=True),
            #)
            if torch.cuda.device_count() > 1:
                model.encoder = torch.nn.DataParallel(model.encoder)
            model = model.cuda()
            criterion = criterion.cuda()
            classifier = classifier.cuda()
            cudnn.benchmark = True

    return model, classifier, criterion, linear_criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""

    #DDP
    #rank = dist.get_rank()
    #device_id = rank % torch.cuda.device_count()

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            #ddp
            #images = images.to(device_id)
            #labels = labels.to(device_id)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.cuda.amp.autocast(enabled=opt.amp):
            features = model(images)
            # TODO: change to 4 features
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if opt.method == 'SupCon':
                loss = criterion(features, labels)
                loss_self = criterion(features)
            elif opt.method == 'SimCLR':
                loss = criterion(features)
                loss_self = criterion(features)
            else:
                raise ValueError('contrastive method not supported: {}'.
                                 format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg, loss_self

def linear_train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""

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
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
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


def linear_validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

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
    print("wait: free the last job memory")
    #time.sleep(30)
    best_acc = 0
    opt = parse_option()
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    # build data loader
    train_loader, linear_train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion, linear_criterion = set_model(opt)


    # build optimizer
    optimizer = set_optimizer(opt, model)
    linear_opt = set_optimizer(opt, classifier)

    init_epoch = 1
    best_loss = 100

    ckpt_path = os.path.join(opt.save_folder, 'best.pth')
    if opt.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        init_epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer'])
        opt = ckpt['opt']
        best_loss = ckpt['loss']

    # training routine
    for epoch in range(init_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        adjust_learning_rate(opt, linear_opt, epoch)
        # train for one epoch
        time1 = time.time()
        loss, loss_self = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()

        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('pretraining_loss', loss, epoch)
        logger.log_value('self_loss', loss_self, epoch)
        logger.log_value('pretraining_learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.eval_freq == 0:
            loss, acc = linear_train(linear_train_loader, model, classifier, linear_criterion,
                              linear_opt, epoch, opt)

            # tensorboard logger
            logger.log_value('training loss', loss, epoch)
            logger.log_value('training accuracy', acc, epoch)

            # eval for one epoch
            loss, val_top1, val_top5 = linear_validate(val_loader, model, classifier, linear_criterion, opt)
            logger.log_value('testing loss', loss, epoch)
            logger.log_value('testing top 1 accuracy', val_top1, epoch)
            logger.log_value('testing top 5 accuracy', val_top5, epoch)
            time3 = time.time()
            print('Train epoch {}, total time after eval {:.2f}, accuracy:{:.2f}'.format(
                epoch, time3 - time1, acc))

        if opt.resume:
            #if loss < best_loss:
            # save the best model
            save_file = os.path.join(
                opt.save_folder, 'best.pth')
            save_model(model, optimizer, opt, epoch, loss, save_file)
            best_loss = loss

        elif epoch % opt.save_freq == 0:

            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, loss, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, loss, save_file)


if __name__ == '__main__':
    main()
