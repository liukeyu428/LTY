import torch as th
import torch.nn.functional as F
import math
import sys
import time
import datetime
import collections
import numpy as np
from .utils import AverageMeter, accuracy, lr_change_over_epoch1, lr_change_over_epoch2

import torch
import torch.nn as nn

import math
import numpy as np

from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

class Kl_loss(nn.Module):
    def __init__(self, stage=4): #  python中的构造函数
        super(Kl_loss, self).__init__()
        self.stages = stage

    def kl_fun(self, alpha): #若标签为 1，则 alp 更加接近 y_new（实际标签），而若标签为 0，则 alp 更加接近模型的预测分布 alpha。将alp与均匀分布做kl散度，接近均匀分布，值为1的类别的证据保持不变，其他类别证据减少。alp=（1，α2，α3，α4）
        beta = torch.tensor(np.ones((1, self.stages)), dtype=torch.float32)
        # beta = torch.tensor(np.ones((1, 5)), dtype=np.float32)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        alpha = alpha.to("cuda")
        beta = beta.to("cuda")
        dg0 = dg0.to("cuda")
        dg1 = dg1.to("cuda")
        lnB = lnB.to("cuda")
        lnB_uni = lnB_uni.to("cuda")
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl

    # Calculate the loss
    def forward(self, y_new, alpha, global_step, annealing_step):
        #alpha = alpha*class_weights
        S = torch.sum(alpha, dim=1, keepdim=True)
        # E = alpha - 1
        m = alpha / S
        A = torch.sum((y_new - m) ** 2, dim=1, keepdim=True)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        num1 = 1.0
        num2 = global_step / annealing_step
        annealing_coef = min(num1, num2)   #退火系数
        alp = alpha * (1 - y_new) + y_new
        C = annealing_coef * self.kl_fun(alp)
        loss = (A + B) + C
        return loss.mean()
def train1(dataLoader, model, device, opt, cls_criterion1, cls_criterion2,
                    start_time, epoch, args): #无分支训练
    print('--------------------------Start training At Epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)
    cls_criterion1 = cls_criterion1.to(device)
    cls_criterion2 = cls_criterion2.to(device)
    source_id_list = args.source_id_list
    target_id = args.target_id
    num_source = len(source_id_list)
    dict_log = {'loss': AverageMeter(), 'last_cls_loss': AverageMeter(), 'acc': AverageMeter()}
    for i in source_id_list:
        dict_log[i] = {'s_cls_loss': AverageMeter(), 't_cls_loss': AverageMeter(),
                       's_acc': AverageMeter(), 't_acc': AverageMeter()}
    if 1 == args.adjust_lr:
        lr_change_over_epoch1(opt, args.lr, epoch, args.epochs)
    elif 2 == args.adjust_lr:
        lr_change_over_epoch2(opt, args.lr, epoch)
    model.train()
    for step, (features, labels) in enumerate(dataLoader):
        batch_size_s = len(features) // (num_source + 1)#每一个受试数据的特征长度
        features = features.to(device)
        labels = labels[:, 0].to(device)
        labels_list = [labels[i * batch_size_s: (i + 1) * batch_size_s] for i in range(num_source + 1)]
        labels = labels_list[num_source]
        labels = F.one_hot(labels)
        opt.zero_grad()

        cls = model(features, is_target_only=False)#8,8的列表，代表源域，目标域，64,4的向量代表目标的分类结果
        loss = cls_criterion1(labels, cls + 1, epoch + 1, 10 * (step + 1))

        if not math.isfinite(loss.item()):
            print("Loss is {} at step{}/{}, stopping training.".format(loss.item(), step, epoch))
            print(loss.item())
            sys.exit(1)
        dict_log['loss'].update(loss.item(), len(features))
        loss.backward()
        opt.step()
        acc = accuracy(cls.detach(), labels_list[num_source].detach())[0]
        dict_log['acc'].update(acc.item(), len(cls))
        if 0 == (step + 1) % args.print_freq:
            lr = list(opt.param_groups)[0]['lr']
            now_time = time.time() - start_time
            et = str(datetime.timedelta(seconds=now_time))[:-7]

            print_information = 'epoch:{}/{}\ttime consumption:{}\tstep:{}/{}\tlr:{}\t'.format(
                epoch + 1, args.epochs, et, step, len(dataLoader), lr)




            loss_info = "{}(val/avg):{:.3f}/{:.3f}\n".format('all_loss',
                                                             dict_log['loss'].val, dict_log['loss'].avg)
            print_information += loss_info
            loss_info = "{}(val/avg):{:.3f}/{:.3f}\n".format('acc',
                                                             dict_log['acc'].val, dict_log['acc'].avg)
            print_information += loss_info
            print(print_information)

    print('--------------------------End training At Epoch:{}--------------------------'.format(epoch + 1))
def train_one_epoch(dataLoader, model, device, opt, cls_criterion1, cls_criterion2,
                    start_time, epoch, args): #八分支最终模型训练
    print('--------------------------Start training At Epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)
    cls_criterion1 = cls_criterion1.to(device)
    cls_criterion2 = cls_criterion2.to(device)
    source_id_list = args.source_id_list
    target_id = args.target_id
    num_source = len(source_id_list)
    branch = 8
    dict_log = {'loss': AverageMeter(), 'last_cls_loss': AverageMeter(), 'acc': AverageMeter()}
    for i in source_id_list:
        dict_log[i] = {'s_cls_loss': AverageMeter(), 't_cls_loss': AverageMeter(),
                       's_acc': AverageMeter(), 't_acc': AverageMeter()}

    if 1 == args.adjust_lr:
        lr_change_over_epoch1(opt, args.lr, epoch, args.epochs)
    elif 2 == args.adjust_lr:
        lr_change_over_epoch2(opt, args.lr, epoch)
    model.train()
    for step, (features, labels) in enumerate(dataLoader):
        batch_size_s = len(features) // (num_source + 1)#每一个受试数据的特征长度
        features = features.to(device)
        labels = labels[:, 0].to(device)
        labels_list = [labels[i * batch_size_s: (i + 1) * batch_size_s] for i in range(num_source + 1)]
        opt.zero_grad()
        t_labels = F.one_hot(labels_list[num_source])

        s_logits_list, t_logits_list, cls = model(features,is_target_only=False)#8,8的列表，代表源域，目标域，64,4的向量代表目标的分类结果
        #print(len(s_logits_list), len(t_logits_list), len(cls))
        s_cls_loss_list = []
        s_cls_acc_list = []
        t_cls_loss_list = []
        t_cls_acc_list = []
        loss1 = 0
        for i, (s_logits, t_logits, s_labels) in enumerate(zip(s_logits_list, t_logits_list, labels_list[:num_source])):#计算每个分支的源域和目标域的交叉熵loss和准确率并写到日志
            #s_cls_loss_list.append(cls_criterion1(s_logits,s_labels ))
            new_labels = F.one_hot(s_labels)
            s_cls_loss_list.append(cls_criterion1(new_labels,s_logits+1,epoch+1,10 * (step + 1)))
            dict_log[source_id_list[i]]['s_cls_loss'].update(s_cls_loss_list[i].item(), len(s_logits))
            s_cls_acc_list.append(accuracy(s_logits.detach(), s_labels.detach())[0])
            dict_log[source_id_list[i]]['s_acc'].update(s_cls_acc_list[i].item(), len(s_logits))
           #t_cls_loss_list.append(cls_criterion1(t_logits, labels_list[num_source] ))
            #new_label_list=[labels[i * batch_size_s: (i + 1) * batch_size_s] for i in range(num_source + 1)]
            #new_label_list[num_source] = F.one_hot(labels_list[num_source])
            t_cls_loss_list.append(cls_criterion1(t_labels,t_logits+1, epoch+1,10 * (step + 1)))
            dict_log[source_id_list[i]]['t_cls_loss'].update(t_cls_loss_list[i].item(), len(labels_list[num_source]))
            t_cls_acc_list.append(accuracy(t_logits.detach(), labels_list[num_source].detach())[0])
            dict_log[source_id_list[i]]['t_acc'].update(t_cls_acc_list[i].item(), len(labels_list[num_source]))

            loss_LS = (s_cls_loss_list[i]) / (8 / branch)
            a = s_cls_loss_list[i]
            loss_LT = t_cls_loss_list[i]
            if i == 0:
                loss1 = loss_LS + loss_LT
            else:
                loss1 = loss1 + loss_LS + loss_LT

        #classifier_loss = cls_criterion2(th.log(cls), labels_list[num_source].detach())#LT(
        classifier_loss = cls_criterion1(t_labels.detach(),cls+1, epoch+1,10 * (step + 1))
        #loss = 0.5 * (th.stack(s_cls_loss_list).mean() + th.stack(t_cls_loss_list).mean())#LS,LT
        #loss = 0.5 * (th.stack(s_cls_loss_list) + th.stack(t_cls_loss_list))
        loss = (loss1/branch) + classifier_loss
        if not math.isfinite(loss.item()):
            print("Loss is {} at step{}/{}, stopping training.".format(loss.item(), step, epoch))
            print(loss.item())
            sys.exit(1)
        dict_log['last_cls_loss'].update(classifier_loss.item(), len(features))
        dict_log['loss'].update(loss.item(), len(features))
        loss.backward()
        opt.step()
        acc = accuracy(cls.detach(), labels_list[num_source].detach())[0]
        dict_log['acc'].update(acc.item(), len(cls))
        if 0 == (step + 1) % args.print_freq:
            lr = list(opt.param_groups)[0]['lr']
            now_time = time.time() - start_time
            et = str(datetime.timedelta(seconds=now_time))[:-7]

            print_information = 'epoch:{}/{}\ttime consumption:{}\tstep:{}/{}\tlr:{}\t'.format(
                epoch + 1, args.epochs, et, step, len(dataLoader), lr)
            for key in source_id_list:
                value = dict_log[key]
                key = str(key)

                loss_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('s_cls_loss/' + key,
                                                                 value['s_cls_loss'].val, value['s_cls_loss'].avg)
                print_information += loss_info

                acc_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('s_acc/' + key,
                                                                value['s_acc'].val, value['s_acc'].avg)
                print_information += acc_info

                loss_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('t_cls_loss/' + key + '_{}'.format(target_id),
                                                                 value['t_cls_loss'].val, value['t_cls_loss'].avg)
                print_information += loss_info

                acc_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('t_acc/' + key + '_{}'.format(target_id),
                                                                value['t_acc'].val, value['t_acc'].avg)
                print_information += (acc_info + '\n')

            loss_info = "{}(val/avg):{:.3f}/{:.3f}\n".format('classifier_loss',
                                                             dict_log['last_cls_loss'].val,
                                                             dict_log['last_cls_loss'].avg)
            print_information += loss_info

            loss_info = "{}(val/avg):{:.3f}/{:.3f}\n".format('all_loss',
                                                             dict_log['loss'].val, dict_log['loss'].avg)
            print_information += loss_info
            loss_info = "{}(val/avg):{:.3f}/{:.3f}\n".format('acc',
                                                             dict_log['acc'].val, dict_log['acc'].avg)
            print_information += loss_info
            print(print_information)

    print('--------------------------End training At Epoch:{}--------------------------'.format(epoch + 1))

def train_one_epoch1(dataLoader, model, device, opt, cls_criterion1, cls_criterion2,
                    start_time, epoch, args,source_x_list1):  #训练选中的源域
    print('--------------------------Start training At Epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)
    cls_criterion1 = cls_criterion1.to(device)
    cls_criterion2 = cls_criterion2.to(device)
    source_id_list = source_x_list1
    target_id = args.target_id
    num_source = len(source_id_list)
    branch = num_source
    dict_log = {'loss': AverageMeter(), 'last_cls_loss': AverageMeter(), 'acc': AverageMeter()}
    for i in source_id_list:
        dict_log[i] = {'s_cls_loss': AverageMeter(), 't_cls_loss': AverageMeter(),
                       's_acc': AverageMeter(), 't_acc': AverageMeter()}

    if 1 == args.adjust_lr:
        lr_change_over_epoch1(opt, args.lr, epoch, args.epochs)
    elif 2 == args.adjust_lr:
        lr_change_over_epoch2(opt, args.lr, epoch)
    model.train()
    for step, (features, labels) in enumerate(dataLoader):
        batch_size_s = len(features) // (num_source + 1)#每一个受试数据的特征长度
        features = features.to(device)
        labels = labels[:, 0].to(device)
        labels_list = [labels[i * batch_size_s: (i + 1) * batch_size_s] for i in range(num_source + 1)]
        opt.zero_grad()
        t_labels = F.one_hot(labels_list[num_source])

        s_logits_list, t_logits_list, cls = model(features, is_target_only=False)#8,8的列表，代表源域，目标域，64,4的向量代表目标的分类结果
        #print(len(s_logits_list), len(t_logits_list), len(cls))
        s_cls_loss_list = []
        s_cls_acc_list = []
        t_cls_loss_list = []
        t_cls_acc_list = []
        loss1 = 0
        for i, (s_logits, t_logits, s_labels) in enumerate(zip(s_logits_list, t_logits_list, labels_list[:num_source])):#计算每个分支的源域和目标域的交叉熵loss和准确率并写到日志
            #s_cls_loss_list.append(cls_criterion1(s_logits,s_labels ))
            new_labels = F.one_hot(s_labels)
            s_cls_loss_list.append(cls_criterion1(new_labels,s_logits+1,epoch+1,10 * (step + 1)))
            dict_log[source_id_list[i]]['s_cls_loss'].update(s_cls_loss_list[i].item(), len(s_logits))
            s_cls_acc_list.append(accuracy(s_logits.detach(), s_labels.detach())[0])
            dict_log[source_id_list[i]]['s_acc'].update(s_cls_acc_list[i].item(), len(s_logits))
           #t_cls_loss_list.append(cls_criterion1(t_logits, labels_list[num_source] ))
            #new_label_list=[labels[i * batch_size_s: (i + 1) * batch_size_s] for i in range(num_source + 1)]
            #new_label_list[num_source] = F.one_hot(labels_list[num_source])
            t_cls_loss_list.append(cls_criterion1(t_labels,t_logits+1, epoch+1,10 * (step + 1)))
            dict_log[source_id_list[i]]['t_cls_loss'].update(t_cls_loss_list[i].item(), len(labels_list[num_source]))
            t_cls_acc_list.append(accuracy(t_logits.detach(), labels_list[num_source].detach())[0])
            dict_log[source_id_list[i]]['t_acc'].update(t_cls_acc_list[i].item(), len(labels_list[num_source]))

            loss_LS = (s_cls_loss_list[i]) / (8 / branch)
            a = s_cls_loss_list[i]
            loss_LT = t_cls_loss_list[i]
            if i == 0:
                loss1 = loss_LS + loss_LT
            else:
                loss1 = loss1 + loss_LS + loss_LT

        #classifier_loss = cls_criterion2(th.log(cls), labels_list[num_source].detach())#LT(
        classifier_loss = cls_criterion1(t_labels.detach(),cls+1, epoch+1,10 * (step + 1))
        #loss = 0.5 * (th.stack(s_cls_loss_list).mean() + th.stack(t_cls_loss_list).mean())#LS,LT
        #loss = 0.5 * (th.stack(s_cls_loss_list) + th.stack(t_cls_loss_list))
        loss = (loss1/branch) + classifier_loss
        if not math.isfinite(loss.item()):
            print("Loss is {} at step{}/{}, stopping training.".format(loss.item(), step, epoch))
            print(loss.item())
            sys.exit(1)
        dict_log['last_cls_loss'].update(classifier_loss.item(), len(features))
        dict_log['loss'].update(loss.item(), len(features))
        loss.backward()
        opt.step()
        acc = accuracy(cls.detach(), labels_list[num_source].detach())[0]
        dict_log['acc'].update(acc.item(), len(cls))
        if 0 == (step + 1) % args.print_freq:
            lr = list(opt.param_groups)[0]['lr']
            now_time = time.time() - start_time
            et = str(datetime.timedelta(seconds=now_time))[:-7]

            print_information = 'epoch:{}/{}\ttime consumption:{}\tstep:{}/{}\tlr:{}\t'.format(
                epoch + 1, args.epochs, et, step, len(dataLoader), lr)
            for key in source_id_list:
                value = dict_log[key]
                key = str(key)

                loss_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('s_cls_loss/' + key,
                                                                 value['s_cls_loss'].val, value['s_cls_loss'].avg)
                print_information += loss_info

                acc_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('s_acc/' + key,
                                                                value['s_acc'].val, value['s_acc'].avg)
                print_information += acc_info

                loss_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('t_cls_loss/' + key + '_{}'.format(target_id),
                                                                 value['t_cls_loss'].val, value['t_cls_loss'].avg)
                print_information += loss_info

                acc_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('t_acc/' + key + '_{}'.format(target_id),
                                                                value['t_acc'].val, value['t_acc'].avg)
                print_information += (acc_info + '\n')

            loss_info = "{}(val/avg):{:.3f}/{:.3f}\n".format('classifier_loss',
                                                             dict_log['last_cls_loss'].val,
                                                             dict_log['last_cls_loss'].avg)
            print_information += loss_info

            loss_info = "{}(val/avg):{:.3f}/{:.3f}\n".format('all_loss',
                                                             dict_log['loss'].val, dict_log['loss'].avg)
            print_information += loss_info
            loss_info = "{}(val/avg):{:.3f}/{:.3f}\n".format('acc',
                                                             dict_log['acc'].val, dict_log['acc'].avg)
            print_information += loss_info
            print(print_information)

    print('--------------------------End training At Epoch:{}--------------------------'.format(epoch + 1))


def train1_one_epoch(dataLoader, model, device, opt, cls_criterion1, cls_criterion2,
                    start_time, epoch, args,source_x_list1): #跨被试
    print('--------------------------Start training At Epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)
    cls_criterion1 = cls_criterion1.to(device)
    cls_criterion2 = cls_criterion2.to(device)
    source_id_list = source_x_list1
    target_id = args.target_id
    num_source = len(source_id_list)
    branch = num_source
    dict_log = {'loss': AverageMeter(), 'last_cls_loss': AverageMeter(), 'acc': AverageMeter()}
    for i in source_id_list:
        dict_log[i] = {'s_cls_loss': AverageMeter(), 't_cls_loss': AverageMeter(),
                       's_acc': AverageMeter(), 't_acc': AverageMeter()}

    if 1 == args.adjust_lr:
        lr_change_over_epoch1(opt, args.lr, epoch, args.epochs)
    elif 2 == args.adjust_lr:
        lr_change_over_epoch2(opt, args.lr, epoch)
    model.train()
    for step, (features, labels) in enumerate(dataLoader):
        batch_size_s = len(features) // (num_source + 1)  # 每一个受试数据的特征长度
        features = features.to(device)
        labels = labels[:, 0].to(device)
        labels_list = [labels[i * batch_size_s: (i + 1) * batch_size_s] for i in range(num_source + 1)]
        opt.zero_grad()
        t_labels = F.one_hot(labels_list[num_source])

        s_logits_list = model(features, is_target_only=False)  # 8,8的列表，代表源域，目标域，64,4的向量代表目标的分类结果
        # print(len(s_logits_list), len(t_logits_list), len(cls))
        s_cls_loss_list = []
        s_cls_acc_list = []
        t_cls_loss_list = []
        t_cls_acc_list = []
        loss1 = 0
        for i, (s_logits, s_labels) in enumerate(
                zip(s_logits_list, labels_list[:num_source])):  # 计算每个分支的源域和目标域的交叉熵loss和准确率并写到日志
            # s_cls_loss_list.append(cls_criterion1(s_logits,s_labels ))
            new_labels = F.one_hot(s_labels)
            s_cls_loss_list.append(cls_criterion1(new_labels, s_logits + 1, epoch + 1, 10 * (step + 1)))
            dict_log[source_id_list[i]]['s_cls_loss'].update(s_cls_loss_list[i].item(), len(s_logits))
            s_cls_acc_list.append(accuracy(s_logits.detach(), s_labels.detach())[0])
            dict_log[source_id_list[i]]['s_acc'].update(s_cls_acc_list[i].item(), len(s_logits))
            # t_cls_loss_list.append(cls_criterion1(t_logits, labels_list[num_source] ))
            # new_label_list=[labels[i * batch_size_s: (i + 1) * batch_size_s] for i in range(num_source + 1)]
            # new_label_list[num_source] = F.one_hot(labels_list[num_source])

            loss_LS = (s_cls_loss_list[i]) / (8 / branch)
            a = s_cls_loss_list[i]
            if i == 0:
                loss1 = loss_LS
            else:
                loss1 = loss1 + loss_LS

        # classifier_loss = cls_criterion2(th.log(cls), labels_list[num_source].detach())#LT(
        #classifier_loss = cls_criterion1(t_labels.detach(), cls + 1, epoch + 1, 10 * (step + 1))
        # loss = 0.5 * (th.stack(s_cls_loss_list).mean() + th.stack(t_cls_loss_list).mean())#LS,LT
        # loss = 0.5 * (th.stack(s_cls_loss_list) + th.stack(t_cls_loss_list))
        loss = (loss1 / branch)
        if not math.isfinite(loss.item()):
            print("Loss is {} at step{}/{}, stopping training.".format(loss.item(), step, epoch))
            print(loss.item())
            sys.exit(1)
        #dict_log['last_cls_loss'].update(classifier_loss.item(), len(features))
        dict_log['loss'].update(loss.item(), len(features))
        loss.backward()
        opt.step()
        #acc = accuracy(cls.detach(), labels_list[num_source].detach())[0]
        #dict_log['acc'].update(acc.item(), len(cls))
        if 0 == (step + 1) % args.print_freq:
            lr = list(opt.param_groups)[0]['lr']
            now_time = time.time() - start_time
            et = str(datetime.timedelta(seconds=now_time))[:-7]

            print_information = 'epoch:{}/{}\ttime consumption:{}\tstep:{}/{}\tlr:{}\t'.format(
                epoch + 1, args.epochs, et, step, len(dataLoader), lr)
            for key in source_id_list:
                value = dict_log[key]
                key = str(key)

                loss_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('s_cls_loss/' + key,
                                                                 value['s_cls_loss'].val, value['s_cls_loss'].avg)
                print_information += loss_info

                acc_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('s_acc/' + key,
                                                                value['s_acc'].val, value['s_acc'].avg)


                print_information += (acc_info + '\n')


            print(print_information)

    print('--------------------------End training At Epoch:{}--------------------------'.format(epoch + 1))


def evaluate(dataLoader, model, device, cls_criterion, start_time, epoch, args): #对无分支进行测试
    print('--------------------------Start Evaluate At Epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)
    cls_criterion = cls_criterion.to(device)

    target_id = args.target_id
    dict_log = {'loss': AverageMeter(), 'acc': AverageMeter(), 'together_acc': AverageMeter()}
    for i in args.source_id_list:
        dict_log[i] = {'t_acc': AverageMeter()}
    model.eval()
    for step, (features, labels) in enumerate(dataLoader):
        features = features.to(device)
        labels = labels[:, 0].to(device)
        labels1 = F.one_hot(labels)
        with th.no_grad():
            _, _, preds = model(features)
            loss = cls_criterion(labels1, preds + 1, epoch + 1, 10 * (step + 1))
            if len(loss.shape) > 0:
                loss = loss.mean()

        dict_log['loss'].update(loss.item(), len(features))
        acc = accuracy(preds.detach(), labels.detach())[0]
        # acc = accuracy(all_preds_list[-1].detach(), labels.detach())[0]
        dict_log['acc'].update(acc.item(), len(features))
        if (step + 1) == len(dataLoader):
            now_time = time.time() - start_time
            et = str(datetime.timedelta(seconds=now_time))[:-7]

            print_information = 'epoch:{}/{}\ttime consumption:{}\tstep:{}/{}\t\n'.format(
                epoch + 1, args.epochs, et, step, len(dataLoader))

            loss_info = "id:{}\t{}(val/avg):{:.3f}/{:.3f}\t{}(val/avg):{:.3f}/{:.3f}\n ".format(target_id,
                                                                                                'loss',
                                                                                                dict_log['loss'].val,
                                                                                                dict_log['loss'].avg,
                                                                                                'acc',
                                                                                                dict_log['acc'].val,
                                                                                                dict_log['acc'].avg)
            print_information += loss_info
            print(print_information)

    print('--------------------------End Evaluate At Epoch:{}--------------------------'.format(epoch + 1))
    return dict_log['acc'].avg, dict_log['loss'].avg

def evaluate_other(dataLoader, model, device, cls_criterion, start_time, epoch, args):#无分支训练后的模型对源域进行测试，找到相似源
    print('--------------------------Start Evaluate At Epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)
    cls_criterion = cls_criterion.to(device)
    source_id_list = args.source_id_list
    num_source = len(source_id_list)
    target_id = args.target_id
    dict_log = {'loss': AverageMeter(), 'acc': AverageMeter(), 'together_acc': AverageMeter()}
    for i in source_id_list:
        dict_log[i] = {'s_cls_loss': AverageMeter(),
                       's_acc': AverageMeter() }
    model.eval()
    for step, (features, labels) in enumerate(dataLoader):
        batch_size_s = len(features) // (num_source + 1)  # 每一个受试数据的特征长度
        features = features.to(device)
        labels = labels[:, 0].to(device)
        labels_list = [labels[i * batch_size_s: (i + 1) * batch_size_s] for i in range(num_source + 1)]
        labels = labels_list[num_source]
        labels1 = F.one_hot(labels)
        with th.no_grad():
            s_cls_loss_list = []
            s_cls_acc_list = []
            source_cls = model(features,is_target_only=True, is_best = True)
            a = source_cls[num_source]
            b = labels_list[num_source]
            for i, (s_logits, s_labels) in enumerate(zip(source_cls, labels_list)):
                new_labels = F.one_hot(s_labels)
                if i >= num_source:
                    loss = cls_criterion(new_labels, s_logits + 1, epoch + 1, 10 * (step + 1))
                    dict_log['loss'].update(loss.item(), len(features))
                    break
                # 计算每个分支的源域和目标域的交叉熵loss和准确率并写到日志
                s_cls_loss_list.append(cls_criterion(new_labels, s_logits + 1, epoch + 1, 10 * (step + 1)))
                dict_log[source_id_list[i]]['s_cls_loss'].update(s_cls_loss_list[i].item(), len(s_logits))
                s_cls_acc_list.append(accuracy(s_logits.detach(), s_labels.detach())[0])
                dict_log[source_id_list[i]]['s_acc'].update(s_cls_acc_list[i].item(), len(s_logits))
            #loss = cls_criterion(labels1,source_cls[num_source],epoch + 1, 10 * (step + 1))
            #dict_log['loss'].update(loss.item(), len(features))
        acc = accuracy(source_cls[num_source].detach(), labels.detach())[0]
        dict_log['acc'].update(acc.item(), len(features))
        if (step + 1) == len(dataLoader):
            now_time = time.time() - start_time
            et = str(datetime.timedelta(seconds=now_time))[:-7]

            print_information = 'epoch:{}/{}\ttime consumption:{}\tstep:{}/{}\t\n'.format(
                epoch + 1, args.epochs, et, step, len(dataLoader))

            for key in source_id_list:
                value = dict_log[key]
                key = str(key)

                loss_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('s_cls_loss/' + key,
                                                                 value['s_cls_loss'].val, value['s_cls_loss'].avg)
                print_information += loss_info

                acc_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('s_acc/' + key,
                                                                value['s_acc'].val, value['s_acc'].avg)
                print_information += (acc_info + '\n')

            loss_info = "id:{}\t{}(val/avg):{:.3f}/{:.3f}\t{}(val/avg):{:.3f}/{:.3f}\n ".format(target_id,
                                                                                                'loss',
                                                                                                dict_log['loss'].val,
                                                                                                dict_log['loss'].avg,
                                                                                                'acc',
                                                                                                dict_log['acc'].val,
                                                                                                dict_log['acc'].avg)
            print_information += loss_info
            print(print_information)

    print('--------------------------End Evaluate At Epoch:{}--------------------------'.format(epoch + 1))
    return dict_log['acc'].avg, dict_log['loss'].avg
def evaluate_one_epoch(dataLoader, model, device, cls_criterion, start_time, epoch, args):#八分支最终模型测试
    print('--------------------------Start Evaluate At Epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)
    cls_criterion = cls_criterion.to(device)
    branch_acc_dict = {}
    target_id = args.target_id
    dict_log = {'loss': AverageMeter(), 'acc': AverageMeter(), 'together_acc': AverageMeter()}
    for i in args.source_id_list:
        dict_log[i] = {'t_acc': AverageMeter()}

    all_preds = []
    all_labels = []


    model.eval()
    for step, (features, labels) in enumerate(dataLoader):
        features = features.to(device)
        labels = labels[:, 0].to(device)
        new_labels = F.one_hot(labels)
        with th.no_grad():
            _, all_preds_list, preds = model(features)
            loss = cls_criterion(new_labels.detach(),preds+1, epoch+1,10 * (step + 1))
            if len(loss.shape) > 0:
                loss = loss.mean()

        dict_log['loss'].update(loss.item(), len(features))
        acc = accuracy(preds.detach(), labels.detach())[0]
        # acc = accuracy(all_preds_list[-1].detach(), labels.detach())[0]
        dict_log['acc'].update(acc.item(), len(features))

        predicted = torch.argmax(preds, dim=1)#计算
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


        for i in range(len(args.source_id_list)):
            acc = accuracy(all_preds_list[i].detach(), labels.detach())[0]
            dict_log[args.source_id_list[i]]['t_acc'].update(acc.item(), len(features))
            #先给各分支的acc排序，排好之后去掉其中差的分支再输入模型进行测试
        if len(all_preds_list) > len(args.source_id_list):
            acc = accuracy(all_preds_list[-1].detach(), labels.detach())[0]
            dict_log['together_acc'].update(acc.item(), len(features))
        if (step + 1) == len(dataLoader):
            now_time = time.time() - start_time
            et = str(datetime.timedelta(seconds=now_time))[:-7]

            print_information = 'epoch:{}/{}\ttime consumption:{}\tstep:{}/{}\t\n'.format(
                epoch + 1, args.epochs, et, step, len(dataLoader))
            for key in args.source_id_list:
                value = dict_log[key]
                key = str(key)
                acc_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('t_acc/' + key + '_{}'.format(target_id),
                                                                value['t_acc'].val, value['t_acc'].avg)
                print_information += (acc_info + '\n')
            if len(all_preds_list) > len(args.source_id_list):
                acc_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('t_acc/{}'.format(target_id),
                                                                dict_log['together_acc'].val,
                                                                dict_log['together_acc'].avg)
                print_information += acc_info + '\n'

            kappa = cohen_kappa_score(all_labels, all_preds)#计算kappa
            print_information += f"Kappa Score: {kappa:.4f}\n"

            loss_info = "id:{}\t{}(val/avg):{:.3f}/{:.3f}\t{}(val/avg):{:.3f}/{:.3f}\n ".format(target_id,
                                                                                                'loss',
                                                                                                dict_log['loss'].val,
                                                                                                dict_log['loss'].avg,
                                                                                                'acc',
                                                                                                dict_log['acc'].val,
                                                                                                dict_log['acc'].avg)
            print_information += loss_info
            print(print_information)

    print('--------------------------End Evaluate At Epoch:{}--------------------------'.format(epoch + 1))
    return dict_log['acc'].avg, dict_log['loss'].avg, all_labels, all_preds

def evaluate_one_epoch1(dataLoader, model, device, cls_criterion, start_time, epoch, args,source_x_list1):#选定其中某些源域进行测试
    print('--------------------------Start Evaluate At Epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)
    cls_criterion = cls_criterion.to(device)

    target_id = args.target_id
    dict_log = {'loss': AverageMeter(), 'acc': AverageMeter(), 'together_acc': AverageMeter()}
    for i in source_x_list1:
        dict_log[i] = {'t_acc': AverageMeter()}
    all_preds = []
    all_labels = []
    model.eval()
    for step, (features, labels) in enumerate(dataLoader):
        features = features.to(device)
        labels = labels[:, 0].to(device)
        new_labels = F.one_hot(labels)
        with th.no_grad():
            _, all_preds_list, preds = model(features)
            loss = cls_criterion(new_labels.detach(),preds+1, epoch+1,10 * (step + 1))
            if len(loss.shape) > 0:
                loss = loss.mean()

        dict_log['loss'].update(loss.item(), len(features))
        acc = accuracy(preds.detach(), labels.detach())[0]
        # acc = accuracy(all_preds_list[-1].detach(), labels.detach())[0]
        dict_log['acc'].update(acc.item(), len(features))

        predicted = torch.argmax(preds, dim=1)  # 计算
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for i in range(len(source_x_list1)):
            acc = accuracy(all_preds_list[i].detach(), labels.detach())[0]
            dict_log[source_x_list1[i]]['t_acc'].update(acc.item(), len(features))
        if len(all_preds_list) > len(source_x_list1):
            acc = accuracy(all_preds_list[-1].detach(), labels.detach())[0]
            dict_log['together_acc'].update(acc.item(), len(features))
        if (step + 1) == len(dataLoader):
            now_time = time.time() - start_time
            et = str(datetime.timedelta(seconds=now_time))[:-7]

            print_information = 'epoch:{}/{}\ttime consumption:{}\tstep:{}/{}\t\n'.format(
                epoch + 1, args.epochs, et, step, len(dataLoader))
            for key in source_x_list1:
                value = dict_log[key]
                key = str(key)
                acc_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('t_acc/' + key + '_{}'.format(target_id),
                                                                value['t_acc'].val, value['t_acc'].avg)
                print_information += (acc_info + '\n')
            if len(all_preds_list) > len(source_x_list1):
                acc_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('t_acc/{}'.format(target_id),
                                                                dict_log['together_acc'].val,
                                                                dict_log['together_acc'].avg)
                print_information += acc_info + '\n'

            kappa = cohen_kappa_score(all_labels, all_preds)  # 计算kappa
            print_information += f"Kappa Score: {kappa:.4f}\n"


            loss_info = "id:{}\t{}(val/avg):{:.3f}/{:.3f}\t{}(val/avg):{:.3f}/{:.3f}\n ".format(target_id,
                                                                                                'loss',
                                                                                                dict_log['loss'].val,
                                                                                                dict_log['loss'].avg,
                                                                                                'acc',
                                                                                                dict_log['acc'].val,
                                                                                                dict_log['acc'].avg)
            print_information += loss_info
            print(print_information)

    print('--------------------------End Evaluate At Epoch:{}--------------------------'.format(epoch + 1))
    return dict_log['acc'].avg, dict_log['loss'].avg,all_labels, all_preds



