import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from unet import UNet
import joint_transforms
from config import msra10k_path
from datasets0 import ImageFolder
from misc import AvgMeter, check_mkdir
from model import R3Net
from torch.backends import cudnn

cudnn.benchmark = True

torch.manual_seed(2018)
torch.cuda.set_device(2)

ckpt_path = './ckpt'
exp_name = 'R3Net'

args = {
    'iter_num': 40000,
    'train_batch_size': 10,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': ''
}

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(300),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

train_set = ImageFolder(msra10k_path, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True)

criterion = nn.BCEWithLogitsLoss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    net = UNet().cuda().train()
    #print 'load pretrained model from R3Net:6000.pth'
    #net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, '6000' + '.pth')))
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print 'training resumes from ' + args['snapshot']
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, loss0_record, loss1_record, loss2_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss3_record, loss4_record, loss5_record, loss6_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()
            outputs0 = net(inputs)
            loss0 = criterion(outputs0, labels)


            total_loss = loss0
            total_loss.backward()
            optimizer.step()

            total_loss_record.update(total_loss.data[0], batch_size)
            loss0_record.update(loss0.data[0], batch_size)


            curr_iter += 1

            log = '[iter %d], [total loss %.5f], [lr %.13f]' % \
                  (curr_iter, total_loss_record.avg,
                   optimizer.param_groups[1]['lr'])
            print log
            open(log_path, 'a').write(log + '\n')

            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d_unet.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim_unet.pth' % curr_iter))
                return


if __name__ == '__main__':
    main()
