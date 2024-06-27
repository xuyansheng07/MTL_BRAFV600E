import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.autograd import Variable
import csv
import argparse
import os
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils import EarlyStopping
from utils import MyData
from MTL_BRAFV600E import MTL_BRAFV600E
from utils import Augmentation
from utils import lossfunction

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# 设置随机种子
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default=r'./Data/brafv600E',
                    type=str, help='dataset root dir')
parser.add_argument('--store_root', default=r'./Weights', type=str, help='dataset root dir')
parser.add_argument('--max_epoch', default=1000, type=int, help='max_epoch')
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
parser.add_argument('--mixup', default=True, type=bool, help='mixup')
parser.add_argument('--patience', default=20, type=int, help='earlystoppingpatience')


def train(experiment: int, opt):
    data_root = opt.data_root
    store_root = opt.store_root
    n_epochs = opt.max_epoch
    batch_size = opt.batchsize
    is_mixup = opt.mixup
    patience = opt.patience

    print('setting up dataloaders...')
    data_transform = {
        "train": transforms.Compose([transforms.Resize([255, 255]),
                                     transforms.RandomCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomAffine(degrees=30, translate=(0, 0.2), scale=(0.9, 1),
                                                             shear=(6, 9), fillcolor=66),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                     ]),
        "val": transforms.Compose([transforms.Resize([224, 224]),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                   ])}

    train_dataset = MyData(r'./Data/train.csv',
                           data_root,
                           transform=data_transform["train"])

    val_dataset = MyData(r'./Data/val.csv',
                         data_root,
                         transform=data_transform["val"])

    test_dataset = MyData(r'./Data/test.csv',
                          data_root,
                          transform=data_transform["val"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    test_num = len(test_dataset)

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    # build model architecture
    print('building the model and loss...')
    net = MTL_BRAFV600E()
    net = nn.DataParallel(net).cuda()

    weight_p, bias_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    loss_function = lossfunction.FocalLoss(gamma=2)
    awl = lossfunction.AutomaticWeightedLoss(6).cuda()

    # build optimization tools
    print('building parameter...')
    patience = patience
    times = 1
    experiment = str(experiment)
    exp_dir = Path(store_root) / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(exp_dir) / 'MTL_BRAFV600E.pth'
    early_stopping = EarlyStopping.EarlyStopping(patience=patience, verbose=True, save_path=save_path)
    val_loss_min = np.Inf

    # start training!
    print('setup complete, start training...')
    writer = SummaryWriter(log_dir=exp_dir)
    for epoch in range(n_epochs):
        train_losses = []
        valid_losses = []

        train_losses_0 = []
        train_losses_1 = []
        train_losses_2 = []
        train_losses_3 = []
        train_losses_4 = []
        train_losses_5 = []


        valid_losses_0 = []
        valid_losses_1 = []
        valid_losses_2 = []
        valid_losses_3 = []
        valid_losses_4 = []
        valid_losses_5 = []


        if epoch < 21:
            optimizer = optim.SGD([{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}], lr=1e-5, momentum=0.95,
                                  weight_decay=0.0005)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

        elif times == 1:
            optimizer = optim.AdamW([{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}], lr=1e-4,
                                    weight_decay=1e-3)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
        # train
        net.train()
        running_loss = 0.0
        train_acc_0 = 0.0
        train_acc_1 = 0.0
        train_acc_2 = 0.0
        train_acc_3 = 0.0
        train_acc_4 = 0.0
        train_acc_5 = 0.0


        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            labels = labels.t().long() - 1
            if is_mixup:
                inputs, targets_a, targets_b, lam = Augmentation.mixup_data(images.cuda(), labels.cuda(), alpha=0.5)
                inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
                logits = net(inputs)
                optimizer.zero_grad()

                loss0 = Augmentation.mixup_criterion(loss_function, logits[0], targets_a[0], targets_b[0], lam)
                loss1 = Augmentation.mixup_criterion(loss_function, logits[1], targets_a[1], targets_b[1], lam)
                loss2 = Augmentation.mixup_criterion(loss_function, logits[2], targets_a[2], targets_b[2], lam)
                loss3 = Augmentation.mixup_criterion(loss_function, logits[3], targets_a[3], targets_b[3], lam)
                loss4 = Augmentation.mixup_criterion(loss_function, logits[4], targets_a[4], targets_b[4], lam)
                loss5 = Augmentation.mixup_criterion(loss_function, logits[5], targets_a[5], targets_b[5], lam)

                train_losses_0.append(loss0.item())
                train_losses_1.append(loss1.item())
                train_losses_2.append(loss2.item())
                train_losses_3.append(loss3.item())
                train_losses_4.append(loss4.item())
                train_losses_5.append(loss5.item())

            else:
                optimizer.zero_grad()
                logits = net(images.cuda())
                loss0 = loss_function(logits[0], labels[0].cuda())
                loss1 = loss_function(logits[1], labels[1].cuda())
                loss2 = loss_function(logits[2], labels[2].cuda())
                loss3 = loss_function(logits[3], labels[3].cuda())
                loss4 = loss_function(logits[4], labels[4].cuda())
                loss5 = loss_function(logits[5], labels[5].cuda())

                train_losses_0.append(loss0.item())
                train_losses_1.append(loss1.item())
                train_losses_2.append(loss2.item())
                train_losses_3.append(loss3.item())
                train_losses_4.append(loss4.item())
                train_losses_5.append(loss5.item())

            loss = awl(loss0.cuda(), loss1.cuda(), loss2.cuda(), loss3.cuda(), loss4.cuda(),loss5.cuda())
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
                # loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            running_loss += loss.item()
            predict_0 = torch.max(logits[0], dim=1)[1]
            predict_1 = torch.max(logits[1], dim=1)[1]
            predict_2 = torch.max(logits[2], dim=1)[1]
            predict_3 = torch.max(logits[3], dim=1)[1]
            predict_4 = torch.max(logits[4], dim=1)[1]
            predict_5 = torch.max(logits[5], dim=1)[1]


            train_acc_0 += (predict_0 == labels[0].cuda()).sum().item()
            train_acc_1 += (predict_1 == labels[1].cuda()).sum().item()
            train_acc_2 += (predict_2 == labels[2].cuda()).sum().item()
            train_acc_3 += (predict_3 == labels[3].cuda()).sum().item()
            train_acc_4 += (predict_4 == labels[4].cuda()).sum().item()
            train_acc_5 += (predict_5 == labels[5].cuda()).sum().item()


            train_accurate_0 = train_acc_0 / train_num
            train_accurate_1 = train_acc_1 / train_num
            train_accurate_2 = train_acc_2 / train_num
            train_accurate_3 = train_acc_3 / train_num
            train_accurate_4 = train_acc_4 / train_num
            train_accurate_5 = train_acc_5 / train_num


            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
            scheduler.step()
        # validate
        net.eval()
        acc_0 = 0.0  # accumulate accurate number / epoch
        acc_1 = 0.0
        acc_2 = 0.0
        acc_3 = 0.0
        acc_4 = 0.0
        acc_5 = 0.0

        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data
                val_labels = val_labels.t().long() - 1

                outputs = net(val_images.cuda())  # eval model only have last output layer

                val_loss_0 = loss_function(outputs[0], val_labels[0].cuda())
                val_loss_1 = loss_function(outputs[1], val_labels[1].cuda())
                val_loss_2 = loss_function(outputs[2], val_labels[2].cuda())
                val_loss_3 = loss_function(outputs[3], val_labels[3].cuda())
                val_loss_4 = loss_function(outputs[4], val_labels[4].cuda())
                val_loss_5 = loss_function(outputs[5], val_labels[5].cuda())

                val_loss = awl(val_loss_0.cuda(), val_loss_1.cuda(), val_loss_2.cuda(), val_loss_3.cuda(),
                               val_loss_4.cuda(),val_loss_5.cuda())
                valid_losses_0.append(val_loss_0.item())
                valid_losses_1.append(val_loss_1.item())
                valid_losses_2.append(val_loss_2.item())
                valid_losses_3.append(val_loss_3.item())
                valid_losses_4.append(val_loss_4.item())
                valid_losses_5.append(val_loss_5.item())

                valid_losses.append(val_loss.item())

                predict_0 = torch.max(outputs[0], dim=1)[1]
                predict_1 = torch.max(outputs[1], dim=1)[1]
                predict_2 = torch.max(outputs[2], dim=1)[1]
                predict_3 = torch.max(outputs[3], dim=1)[1]
                predict_4 = torch.max(outputs[4], dim=1)[1]
                predict_5 = torch.max(outputs[5], dim=1)[1]


                acc_0 += (predict_0 == val_labels[0].cuda()).sum().item()
                acc_1 += (predict_1 == val_labels[1].cuda()).sum().item()
                acc_2 += (predict_2 == val_labels[2].cuda()).sum().item()
                acc_3 += (predict_3 == val_labels[3].cuda()).sum().item()
                acc_4 += (predict_4 == val_labels[4].cuda()).sum().item()
                acc_5 += (predict_5 == val_labels[5].cuda()).sum().item()



        val_accurate_0 = acc_0 / val_num
        val_accurate_1 = acc_1 / val_num
        val_accurate_2 = acc_2 / val_num
        val_accurate_3 = acc_3 / val_num
        val_accurate_4 = acc_4 / val_num
        val_accurate_5 = acc_5 / val_num


        writer.add_scalar('composition_val_acc', val_accurate_0, epoch)
        writer.add_scalar('echo_val_acc', val_accurate_1, epoch)
        writer.add_scalar('margin_val_acc', val_accurate_2, epoch)
        writer.add_scalar('foci_val_acc', val_accurate_3, epoch)
        writer.add_scalar('shape_val_acc', val_accurate_4, epoch)
        writer.add_scalar('braf_val_acc', val_accurate_5, epoch)

        writer.add_scalar('composition_train_acc', train_accurate_0, epoch)
        writer.add_scalar('echo_train_acc', train_accurate_1, epoch)
        writer.add_scalar('margin_train_acc', train_accurate_2, epoch)
        writer.add_scalar('foci_train_acc', train_accurate_3, epoch)
        writer.add_scalar('shape_train_acc', train_accurate_4, epoch)
        writer.add_scalar('braf_train_acc', train_accurate_5, epoch)

        print('\r[epoch %d] composition:train_loss: %.3f  train_acc_0:%.3f  val_acc_0: %.3f ' % (
        epoch + 1, np.average(train_losses_0), train_accurate_0, val_accurate_0))
        print('\r[epoch %d] echo:train_loss: %.3f  train_acc_1:%.3f  val_acc_1: %.3f ' % (
        epoch + 1, np.average(train_losses_1), train_accurate_1, val_accurate_1))
        print('\r[epoch %d] margin:train_loss: %.3f  train_acc_2:%.3f  val_acc_2: %.3f ' % (
        epoch + 1, np.average(train_losses_2), train_accurate_2, val_accurate_2))
        print('\r[epoch %d] foci:train_loss: %.3f  train_acc_3:%.3f  val_acc_3: %.3f ' % (
        epoch + 1, np.average(train_losses_3), train_accurate_3, val_accurate_3))
        print('\r[epoch %d] shape:train_loss: %.3f  train_acc_4:%.3f  val_acc_4: %.3f ' % (
        epoch + 1, np.average(train_losses_4), train_accurate_4, val_accurate_4))
        print('\r[epoch %d] braf:train_loss: %.3f  train_acc_5:%.3f  val_acc_5: %.3f ' % (
        epoch + 1, np.average(train_losses_5), train_accurate_5, val_accurate_5))

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        writer.add_scalar('train_loss', train_loss, epoch)

        train_loss_composition = np.average(train_losses_0)
        train_loss_echo = np.average(train_losses_1)
        train_loss_margin = np.average(train_losses_2)
        train_loss_foci = np.average(train_losses_3)
        train_loss_shape = np.average(train_losses_4)
        train_loss_braf = np.average(train_losses_5)

        writer.add_scalar('shape_train_loss', train_loss_shape, epoch)
        writer.add_scalar('margin_train_loss', train_loss_margin, epoch)
        writer.add_scalar('braf_train_loss', train_loss_braf, epoch)
        writer.add_scalar('echo_train_loss', train_loss_echo, epoch)
        writer.add_scalar('composition_train_loss', train_loss_composition, epoch)
        writer.add_scalar('foci_train_loss', train_loss_foci, epoch)

        valid_loss = np.average(valid_losses)
        writer.add_scalar('val_loss', valid_loss, epoch)

        val_loss_composition = np.average(valid_losses_0)
        val_loss_echo = np.average(valid_losses_1)
        val_loss_margin = np.average(valid_losses_2)
        val_loss_foci = np.average(valid_losses_3)
        val_loss_shape = np.average(valid_losses_4)
        val_loss_braf = np.average(valid_losses_5)
        writer.add_scalar('shape_val_loss', val_loss_shape, epoch)
        writer.add_scalar('margin_val_loss', val_loss_margin, epoch)
        writer.add_scalar('bm_val_loss', val_loss_braf, epoch)
        writer.add_scalar('echo_val_loss', val_loss_echo, epoch)
        writer.add_scalar('composition_val_loss', val_loss_composition, epoch)
        writer.add_scalar('foci_val_loss', val_loss_foci, epoch)

        epoch_len = len(str(n_epochs))
        print_msg = (f'\r[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)
        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, net)
        if early_stopping.early_stop:
            if times == 0:
                print("Early stopping")
                break
            times -= 1
            early_stopping.counter = 0
            early_stopping.early_stop = False
            net = torch.load(save_path)
            optimizer = optim.SGD([{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}], lr=1e-5, momentum=0.95,
                                  weight_decay=0.0005)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)



    print("starting test.....")

    test_loss = 0.0
    y_0_pred = []
    y_0_true = []

    y_1_pred = []
    y_1_true = []

    y_2_pred = []
    y_2_true = []

    y_3_pred = []
    y_3_true = []

    y_4_pred = []
    y_4_true = []

    y_5_pred = []
    y_5_true = []

    test_acc1 = 0.0
    test_acc2 = 0.0
    test_acc3 = 0.0
    test_acc4 = 0.0
    test_acc5 = 0.0
    test_acc6 = 0.0


    test_accurate_1 = 0.0
    test_accurate_2 = 0.0
    test_accurate_3 = 0.0
    test_accurate_4 = 0.0
    test_accurate_5 = 0.0
    test_accurate_6 = 0.0

    model = torch.load(save_path)
    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            target = target.t().long() - 1
            outputs = model(data.cuda())
            test_loss_0 = loss_function(outputs[0], target[0].cuda())
            test_loss_1 = loss_function(outputs[1], target[1].cuda())
            test_loss_2 = loss_function(outputs[2], target[2].cuda())
            test_loss_3 = loss_function(outputs[3], target[3].cuda())
            test_loss_4 = loss_function(outputs[4], target[4].cuda())
            test_loss_5 = loss_function(outputs[5], target[5].cuda())

            test_loss = awl(test_loss_0.cuda(), test_loss_1.cuda(), test_loss_2.cuda(),
                            test_loss_3.cuda(),
                            test_loss_4.cuda(), test_loss_5.cuda())

            test_loss += test_loss.item() * data.size(0)

            _, pred_0 = torch.max(outputs[0], 1)
            _, pred_1 = torch.max(outputs[1], 1)
            _, pred_2 = torch.max(outputs[2], 1)
            _, pred_3 = torch.max(outputs[3], 1)
            _, pred_4 = torch.max(outputs[4], 1)
            _, pred_5 = torch.max(outputs[5], 1)

            y_0_true.extend(np.array(target[0].cpu()))
            y_0_pred.extend(np.array(pred_0.cpu()))

            y_1_true.extend(np.array(target[1].cpu()))
            y_1_pred.extend(np.array(pred_1.cpu()))

            y_2_true.extend(np.array(target[2].cpu()))
            y_2_pred.extend(np.array(pred_2.cpu()))

            y_3_true.extend(np.array(target[3].cpu()))
            y_3_pred.extend(np.array(pred_3.cpu()))

            y_4_true.extend(np.array(target[4].cpu()))
            y_4_pred.extend(np.array(pred_4.cpu()))

            y_5_true.extend(np.array(target[5].cpu()))
            y_5_pred.extend(np.array(pred_5.cpu()))


            pred_0 = np.array(pred_0.cpu())
            pred_1 = np.array(pred_1.cpu())
            pred_2 = np.array(pred_2.cpu())
            pred_3 = np.array(pred_3.cpu())
            pred_4 = np.array(pred_4.cpu())
            pred_5 = np.array(pred_5.cpu())


            test_acc1 += (pred_0 == target[0].cuda()).sum().item()
            test_acc2 += (pred_1 == target[1].cuda()).sum().item()
            test_acc3 += (pred_2 == target[2].cuda()).sum().item()
            test_acc4 += (pred_3 == target[3].cuda()).sum().item()
            test_acc5 += (pred_4 == target[4].cuda()).sum().item()
            test_acc6 += (pred_5 == target[5].cuda()).sum().item()


    test_accurate_1 = test_acc1 / test_num
    test_accurate_2 = test_acc2 / test_num
    test_accurate_3 = test_acc3 / test_num
    test_accurate_4 = test_acc4 / test_num
    test_accurate_5 = test_acc5 / test_num
    test_accurate_6 = test_acc6 / test_num


    print('\rcomposition_test_acc: %.3f ' % (test_accurate_1))
    print('\recho_test_acc: %.3f ' % (test_accurate_2))
    print('\rmargin_test_acc: %.3f ' % (test_accurate_3))
    print('\rfoci_test_acc: %.3f ' % (test_accurate_4))
    print('\rshape_test_acc: %.3f ' % (test_accurate_5))
    print('\rbraf_test_acc: %.3f ' % (test_accurate_6))



if __name__ == '__main__':
    opt = parser.parse_args()
    for i in range(10):
        print("The result of the"+ i +"-th experiment is as follows:")
        train(i, opt)

