import argparse
import os
import torch
import torch.nn.functional as F
import datetime

from torch.utils.data import DataLoader
from datasets.samplers import CategoriesSampler
from datasets.mini_imagenet import SSLMiniImageNet, MiniImageNet
from datasets.tiered_imagenet import SSLTieredImageNet, TieredImageNet
from datasets.cifarfs import SSLCifarFS, CIFAR_FS
from resnet import resnet12
from util import str2bool, set_gpu, ensure_path, save_checkpoint, count_acc, seed_torch, Averager, euclidean_metric, compute_confidence_interval, normalize, Timer, cos_metric

def get_dataset(args):
    if args.dataset == 'mini':
        trainset = SSLMiniImageNet('train', args)
        valset = MiniImageNet('test', args.size)
        n_cls = 64
        print("=> MiniImageNet...")
    elif args.dataset == 'tiered':
        trainset = SSLTieredImageNet('train', args)
        valset = TieredImageNet('test', args.size)
        n_cls = 351
        print("=> TieredImageNet...")
    elif args.dataset == 'cifarfs':
        trainset = SSLCifarFS('train', args)
        valset = CIFAR_FS('test', args.size)
        n_cls = 64
        print("=> CIFAR-FS...")
    else:
        print("Invalid dataset...")
        exit()
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size,
                                shuffle=True, drop_last=True,
                                num_workers=args.worker, pin_memory=True)

    val_sampler = CategoriesSampler(valset.label, args.test_batch,
                                    args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=args.worker, pin_memory=True)
    return train_loader, val_loader, n_cls

def main(args):
    if args.detail:
        print("=> Self-supervised Training start...")
        print("--------------------------------------------------------")
    ensure_path(args.save_path)

    train_loader, val_loader, n_cls = get_dataset(args)
    # model
    if args.dataset in ['mini', 'tiered']:
        model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_cls).cuda()
    elif args.dataset in ['cifarfs']:
        model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=n_cls).cuda()
    else:
        print("Invalid dataset...")
        exit()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
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

    # check resume point
    checkpoint_file = os.path.join(args.save_path, 'checkpoint.pth.tar')
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        trlog = checkpoint['trlog']
        start_epoch = checkpoint['start_epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cmi[0] = trlog['max_acc']
        print("=> Resume from epoch {} ...".format(start_epoch))

    for epoch in range(start_epoch, args.epochs + 1):

        tl, ta = train(args, train_loader, model, optimizer)
        vl, va, aa, bb = validation(args, val_loader, model)
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

def preprocess_data(data):
    for idxx, img in enumerate(data):
        # 4,3,84,84
        x = img.data[0].unsqueeze(0)
        x90 = img.data[1].unsqueeze(0).transpose(2,3).flip(2)
        x180 = img.data[2].unsqueeze(0).flip(2).flip(3)
        x270 = img.data[3].unsqueeze(0).flip(2).transpose(2,3)
        if idxx <= 0:
            xlist = x
            x90list = x90
            x180list = x180
            x270list = x270
        else:
            xlist = torch.cat((xlist, x), 0)
            x90list = torch.cat((x90list, x90), 0)
            x180list = torch.cat((x180list, x180), 0)
            x270list = torch.cat((x270list, x270), 0)
    # combine
    return torch.cat((xlist, x90list, x180list, x270list), 0).cuda()

def train(args, dataloader, model, optimizer):
    model.train()

    tl = Averager()
    ta = Averager()

    for i, (inputs, target) in enumerate(dataloader, 1):
        target = target.cuda()
        #ssl content
        inputs = preprocess_data(inputs['data'])
        target = target.repeat(4)

        rot_labels = torch.zeros(4*args.batch_size).cuda().long()
        for i in range(4*args.batch_size):
            if i < args.batch_size:
                rot_labels[i] = 0
            elif i < 2*args.batch_size:
                rot_labels[i] = 1
            elif i < 3*args.batch_size:
                rot_labels[i] = 2
            else:
                rot_labels[i] = 3

        _, train_logit, rot_logits = model(inputs, ssl=True)
        rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
        #
        loss_ss = torch.sum(F.binary_cross_entropy_with_logits(
            input=rot_logits, target=rot_labels))
        loss_ce = F.cross_entropy(train_logit, target)
        loss = args.gamma * loss_ss + loss_ce
        acc = count_acc(train_logit, target)

        # result
        tl.add(loss.item())
        ta.add(acc)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return tl.item(), ta.item()

def validation(args, dataloader, model):
    model.eval()

    vl = Averager()
    va = Averager()
    acc_list = []

    for i, batch in enumerate(dataloader, 1):
        data, _ = [_.cuda() for _ in batch]
        p = args.shot * args.way
        data_shot, data_query = data[:p], data[p:]

        proto = model(data_shot, is_feat=args.is_feat)
        proto = proto.reshape(args.shot, args.way, -1).mean(dim=0)

        label = torch.arange(args.way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)

        qf = model(data_query, is_feat=args.is_feat)
        if args.norm:
            proto = normalize(proto)
            qf = normalize(qf)

        if args.distance == 'euc':
            logits = euclidean_metric(qf, proto)
        else:
            logits = cos_metric(qf, proto)
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
    parser.add_argument('--save-path', default='./save/0')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--detail', type=str2bool, nargs='?', default=True)
    # network
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--lr-decay-epochs', type=str, default='60,80')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    # ssl
    parser.add_argument('--gamma', type=float, default=2.0)
    # dataset
    parser.add_argument('--dataset', default='mini', choices=['mini','tiered','cifarfs'])
    parser.add_argument('--size', type=int, default=84)
    parser.add_argument('--worker', type=int, default=8)
    # few-shot
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--test-batch', type=int, default=2000)
    parser.add_argument('--norm', type=str2bool, nargs='?', default=True)
    parser.add_argument('--is-feat', type=str2bool, nargs='?', default=True)
    parser.add_argument('--distance', type=str, default='euc', choices=['cos', 'euc'])
    args = parser.parse_args()
    #
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
