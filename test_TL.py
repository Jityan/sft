import argparse
import os
import torch
import torch.nn.functional as F
import datetime

from torch.utils.data import DataLoader
from datasets.samplers import CategoriesSampler
from datasets.mini_imagenet import MiniImageNet
from datasets.tiered_imagenet import TieredImageNet
from datasets.cifarfs import CIFAR_FS
from resnet import resnet12
from util import str2bool, set_gpu, count_acc, seed_torch, compute_confidence_interval
from model import SFTAttn


def get_dataset(args):
    if args.dataset == 'mini':
        testset = MiniImageNet('test', args.size)
        n_cls = 64
        print("=> MiniImageNet...")
    elif args.dataset == 'tiered':
        testset = TieredImageNet('test', args.size)
        n_cls = 351
        print("=> TieredImageNet...")
    elif args.dataset == 'cifarfs':
        testset = CIFAR_FS('test', args.size)
        n_cls = 64
        print("=> CIFAR-FS...")
    else:
        print("Invalid dataset...")
        exit()

    test_sampler = CategoriesSampler(testset.label, args.test_batch,
                                    args.test_way, args.shot + args.test_query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler,
                            num_workers=args.worker, pin_memory=True)
    return test_loader, n_cls

def main(args):
    loader, n_cls = get_dataset(args)

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

    # check save point
    checkpoint_file = os.path.join(args.save_path, 'max-acc.pth.tar')
    if not os.path.isfile(checkpoint_file):
        print("=> Model not found...")
        exit()
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model'])
    print("=> Model loaded from best epoch", checkpoint['best_epoch'], "...")
    
    encoder1.eval()
    encoder2.eval()
    model.eval()

    acc_list = []

    with torch.no_grad():
        for _, batch in enumerate(loader, 1):
            data, _ = [_.cuda() for _ in batch]
            label = torch.arange(args.test_way).repeat(args.test_query)
            label = label.type(torch.cuda.LongTensor)

            with torch.no_grad():
                data1 = encoder1(data, is_feat=args.is_feat)
                data2 = encoder2(data, is_feat=args.is_feat)

            logits = model(data1, data2)
            acc = count_acc(logits, label)
            acc_list.append(acc*100)

    a, b = compute_confidence_interval(acc_list)
    print("{}-way {}-shot accuracy with 95% interval : {:.2f}±{:.2f}".format(args.test_way, args.shot, a, b))


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    # settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', default='./save/mini_final')
    parser.add_argument('--pretrain-path-1', default='./save/mini_rot')
    parser.add_argument('--pretrain-path-2', default='./save/mini_dist')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=1)
    # fusion
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.5)
    # network
    parser.add_argument('--h-dim', type=int, default=640)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--temperature', type=float, default=1.0)
    # dataset
    parser.add_argument('--dataset', default='mini', choices=['mini','tiered','cifarfs'])
    parser.add_argument('--size', type=int, default=84)
    parser.add_argument('--worker', type=int, default=8)
    # few-shot
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--test-query', type=int, default=15)
    parser.add_argument('--test-batch', type=int, default=2000)
    parser.add_argument('--norm', type=str2bool, nargs='?', default=True)
    parser.add_argument('--is-feat', type=str2bool, nargs='?', default=True)
    parser.add_argument('--distance', type=str, default='euc', choices=['cos', 'euc'])
    args = parser.parse_args()
    
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
