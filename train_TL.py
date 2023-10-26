import argparse
import os
import torch
import torch.nn.functional as F
import datetime
import numpy as np

from torch.utils.data import DataLoader
from datasets.samplers import CategoriesSampler
from datasets.mini_imagenet import MiniImageNet
from datasets.tiered_imagenet import TieredImageNet
from datasets.cifarfs import CIFAR_FS
from resnet import resnet12
from util import str2bool, set_gpu, ensure_path, save_checkpoint, count_acc, seed_torch, Averager, compute_confidence_interval, Timer
from model import SFTAttn


def get_dataset(args):
    from torchvision import transforms
    transform = transforms.Compose([
                    transforms.Resize(args.size),
                    transforms.CenterCrop(args.size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])
    cifar_transform = transforms.Compose([
                    transforms.Resize(args.size),
                    transforms.CenterCrop(args.size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                        np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
                    )
                ])
    if args.dataset == 'mini':
        trainset = MiniImageNet('train', args.size, transform=transform)
        valset = MiniImageNet('test', args.size)
        n_cls = 64
        print("=> MiniImageNet...")
    elif args.dataset == 'tiered':
        trainset = TieredImageNet('train', args.size, transform=transform)
        valset = TieredImageNet('test', args.size)
        n_cls = 351
        print("=> TieredImageNet...")
    elif args.dataset == 'cifarfs':
        trainset = CIFAR_FS('train', args.size, transform=cifar_transform)
        valset = CIFAR_FS('test', args.size)
        n_cls = 64
        print("=> CIFAR-FS...")
    else:
        print("Invalid dataset...")
        exit()
    train_sampler = CategoriesSampler(trainset.label, args.train_batch,
                                    args.train_way, args.shot + args.train_query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                            num_workers=args.worker, pin_memory=True)

    val_sampler = CategoriesSampler(valset.label, args.test_batch,
                                    args.test_way, args.shot + args.test_query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=args.worker, pin_memory=True)
    return train_loader, val_loader, n_cls

def main(args):
    if args.detail:
        print("=> Training start...")
        print("--------------------------------------------------------")
    ensure_path(args.save_path)

    train_loader, val_loader, n_cls = get_dataset(args)
    # model
    if args.dataset in ['mini', 'tiered']:
        encoder1 = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_cls).cuda()
        encoder2 = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_cls).cuda()
    elif args.dataset in ['cifarfs']:
        encoder1 = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=n_cls).cuda()
        encoder2 = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=n_cls).cuda()
    else:
        print("Invalid dataset...")
        exit()
    
    if args.pretrain_path_1:
        checkpoint_file = os.path.join(args.pretrain_path_1, 'max-acc.pth.tar')
        if not os.path.isfile(checkpoint_file):
            print("=> Pretrain model 1 not found...")
            exit()
        checkpoint = torch.load(checkpoint_file)
        encoder1.load_state_dict(checkpoint['model'])
        print("=> Pretrain encoder 1 loaded...")
    
    if args.pretrain_path_2:
        checkpoint_file = os.path.join(args.pretrain_path_2, 'max-acc.pth.tar')
        if not os.path.isfile(checkpoint_file):
            print("=> Pretrain model 2 not found...")
            exit()
        checkpoint = torch.load(checkpoint_file)
        encoder2.load_state_dict(checkpoint['model'])
        print("=> Pretrain encoder 2 loaded...")
    
    model = SFTAttn(args).cuda()

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epochs, gamma=args.lr_decay_rate)
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['train_acc'] = []
    trlog['val_loss'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['best_epoch'] = 0
    start_epoch = 1
    cmi = [0.0, 0.0]
    timer = Timer()


    for epoch in range(start_epoch, args.epochs + 1):

        tl, ta = train(args, train_loader, encoder1, encoder2, model, optimizer)
        vl, va, aa, bb = validation(args, val_loader, encoder1, encoder2, model)
        lr_scheduler.step()

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            trlog['best_epoch'] = epoch
            cmi[0] = aa
            cmi[1] = bb
            # save best model
            save_checkpoint({
                'best_epoch': epoch,
                'model': model.state_dict()
            }, args.save_path, name='max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        # checkpoint saving
        save_checkpoint({
            'start_epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'trlog': trlog
        }, args.save_path)

        ot, ots = timer.measure()
        tt, _ = timer.measure(epoch / args.epochs)

        if args.detail:
            print('Epoch {}/{}: train loss {:.4f} - acc {:.2f}% - val loss {:.4f} - acc {:.2f}% - best acc {:.2f}% - ETA:{}/{}'.format(
                epoch, args.epochs, tl, ta*100, vl, va*100, trlog['max_acc']*100, ots, timer.tts(tt-ot)))

        if epoch >= args.epochs:
            print("Best Epoch is {} with acc={:.2f}±{:.2f}%...".format(trlog['best_epoch'], cmi[0], cmi[1]))
            print("--------------------------------------------------------")
    return


def train(args, dataloader, encoder1, encoder2, model, optimizer):
    encoder1.eval()
    encoder2.eval()
    model.train()

    tl = Averager()
    ta = Averager()

    for i, batch in enumerate(dataloader, 1):
        data, _ = [_.cuda() for _ in batch]

        label = torch.arange(args.train_way).repeat(args.train_query)
        label = label.type(torch.cuda.LongTensor)
        label_aux = torch.arange(args.train_way).repeat(args.shot + args.train_query)
        label_aux = label_aux.type(torch.cuda.LongTensor)

        with torch.no_grad():
            data1 = encoder1(data, is_feat=args.is_feat)
            data2 = encoder2(data, is_feat=args.is_feat)

        logits, reg_logits = model(data1, data2)
        if reg_logits is not None:
            loss = F.cross_entropy(logits, label)
            loss = loss + args.balance * F.cross_entropy(reg_logits, label_aux)
        else:
            loss = F.cross_entropy(logits, label)
        #
        acc = count_acc(logits, label)
        tl.add(loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return tl.item(), ta.item()

def validation(args, dataloader, encoder1, encoder2, model):
    encoder1.eval()
    encoder2.eval()
    model.eval()

    vl = Averager()
    va = Averager()
    acc_list = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader, 1):
            data, _ = [_.cuda() for _ in batch]
            label = torch.arange(args.test_way).repeat(args.test_query)
            label = label.type(torch.cuda.LongTensor)

            data1 = encoder1(data, is_feat=args.is_feat)
            data2 = encoder2(data, is_feat=args.is_feat)

            logits = model(data1, data2)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)
            acc_list.append(acc*100)

    a, b = compute_confidence_interval(acc_list)
    return vl.item(), va.item(), a, b


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    # settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', default='./save/mini_final')
    parser.add_argument('--pretrain-path-1', default='./save/mini_rot')
    parser.add_argument('--pretrain-path-2', default='./save/mini_dist')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--detail', type=str2bool, nargs='?', default=True)
    # network
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--optim', type=str, default='sgd', choices=['adam', 'sgd'])
    parser.add_argument('--lr-decay-epochs', type=str, default='60')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1)
    # fusion
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.5)
    # feat
    parser.add_argument('--balance', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--temperature2', type=float, default=1.0)
    parser.add_argument('--h-dim', type=int, default=640)
    parser.add_argument('--dropout', type=float, default=0.4)
    # dataset
    parser.add_argument('--dataset', default='mini', choices=['mini','tiered','cifarfs'])
    parser.add_argument('--size', type=int, default=84)
    parser.add_argument('--worker', type=int, default=8)
    # few-shot
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--train-query', type=int, default=15)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--test-query', type=int, default=15)
    parser.add_argument('--train-batch', type=int, default=100)
    parser.add_argument('--test-batch', type=int, default=2000)
    parser.add_argument('--norm', type=str2bool, nargs='?', default=True)
    parser.add_argument('--is-feat', type=str2bool, nargs='?', default=True)
    parser.add_argument('--distance', type=str, default='euc', choices=['cos', 'euc'])
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    
    if args.dataset in ['mini', 'tiered']:
        args.size = 84
    elif args.dataset in ['cifarfs']:
        args.size = 32
        args.worker = 0
    else:
        args.size = 28
    # fix seed
    seed_torch(args.seed)
    set_gpu(args.gpu)

    main(args)

    end_time = datetime.datetime.now()
    print("End time :{} total ({})".format(end_time, end_time - start_time))
