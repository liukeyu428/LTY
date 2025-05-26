import torch
import torch as th
import os
import sys
import argparse
import time
import numpy as np
from torch.utils.data import DataLoader

from tools.utils import set_seed, set_save_path, Logger, save, EarlyStopping
from tools.run_tools import  Kl_loss, train1_one_epoch, evaluate_one_epoch1

from models.EEGNet import CEDLCross

from data.bciciv2a_process import load_bciciv2a_data_single_subject
from data.high_gamma_process import load_highgamma_data_single_subject
from data.bciciv2b_process import load_bciciv2b_data_single_subject
from data.sampler import BalanceIDSampler
from data.eegdataset import EEGDataset, EEGDataLoader
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def train(args):
    # ----------------------------------------------environment setting-----------------------------------------------
    set_seed(args.seed)
    args = set_save_path(args.father_path, args)
    sys.stdout = Logger(os.path.join(args.log_path, 'information.txt'))
    start_epoch = 0
    # ------------------------------------------------device setting--------------------------------------------------
    device = 'cuda:0' if th.cuda.is_available() else 'cpu'

    # ------------------------------------------------data setting----------------------------------------------------
    if "high_gamma" in args.data_path:
        args.sub_num = 14
        args.class_num = 4
        load_data = load_highgamma_data_single_subject
    elif "BCICIV_2a" in args.data_path:
        args.sub_num = 9
        args.class_num = 4
        load_data = load_bciciv2a_data_single_subject
    elif "BCICIV_2b" in args.data_path:
        args.sub_num = 9
        args.class_num = 2
        load_data = load_bciciv2b_data_single_subject
    else:
        raise ValueError("only support hipgh_gamma or BCICIV_2a dataset.")
    id_list = [i + 1 for i in range(args.sub_num)] #i=1-9
    source_id_list = []
    source_X_list, source_y_list = [], []
    target_X, target_y, target_test_X, target_test_y = [None] * 4
    for i in id_list:
        if i != args.target_id:
            train_X, train_y, test_source_x, test_source_y= load_data(args.data_path, subject_id=i, to_tensor=False)
            source_id_list.append(i)
            source_X_list.append(train_X)
            source_X_list.append(test_source_x)
            source_y_list.append(train_y)
            source_y_list.append(test_source_y)
        else:
            target_X, target_y, target_test_X, target_test_y = load_data(args.data_path, subject_id=i, to_tensor=False)
    args.source_id_list = source_id_list
    merged_list = []
    merged_listy = []
    for i in range(0, len(source_X_list), 2):
        merged = np.concatenate([source_X_list[i], source_X_list[i + 1]], axis=0)
        mergedy = np.concatenate([source_y_list[i], source_y_list[i + 1]], axis=0)# 在样本维度合并
        merged_list.append(merged)
        merged_listy.append(mergedy)
    source_X_list = merged_list
    source_y_list = merged_listy
    def extract_elements_based_on_indices(list1, list2,list3, n,target):
        """
        根据第一个列表中前 n 个元素的 int 值，从第二个列表中提取对应的张量。

        参数:
        - list1 (list): 一个包含 8 个元组的列表，每个元组包含一个 int 值和一个 float 值。
        - list2 (list): 一个包含 8 个形状为 (288, 22, 1125) 张量的列表。
        - n (int): 指定从 list1 中提取的前 n 个元素。

        返回:
        - extracted_tensors (list): 根据 int 值提取的张量列表。
        """
        # 提取 list1 的前 n 个元素
        top_n_elements = list1[:n]
        extracted_tensors = []
        extracted_labels = []
        i = 0
        # 从这些元素中提取 int 值
        #indices = [elem[0] for elem in top_n_elements]
        # 根据这些 int 值从 list2 中提取对应的张量
        for idx in top_n_elements:
            if idx > target:
                extracted_tensors.append(list2[idx - 2])
                extracted_labels.append(list3[idx - 2])
            else:
                extracted_tensors.append(list2[idx - 1])
                extracted_labels.append(list3[idx - 1])
            i=i+1
        #extracted_tensors = [list2[idx-2] for idx in top_n_elements]
        #extracted_labels = [list3[idx - 2] for idx in top_n_elements]

        return extracted_tensors,list1,extracted_labels
    list1=[2,3,4,5,6,7,8,9]
    args.batch_size = (len(list1)+1)*64
    #source_X_list1,source_id_list,source_y_list1 = extract_elements_based_on_indices(list1, source_X_list, source_y_list,len(list1),args.target_id)
    data_sampler = BalanceIDSampler(source_X_list, target_X, source_y_list, target_y, args.batch_size)#1：8加载batch
    args.batch_size = data_sampler.revised_batch_size
    data_num = data_sampler.__len__()
    train_data = EEGDataset(source_X_list, target_X, source_y_list, target_y, data_num)#改
    trainLoader = DataLoader(train_data, args.batch_size, shuffle=False, sampler=data_sampler, num_workers=8,
                             drop_last=False)
    test_data = EEGDataset(source_X_list=None, target_X=target_test_X, source_y_list=None, target_y=target_test_y)
    testLoader = DataLoader(test_data, args.batch_size // (len(source_id_list) + 1),
                            shuffle=False, num_workers=4, drop_last=False)
    # ------------------------------------------------model setting----------------------------------------------------

    '''backbone = WMB_EEGNet(source_X_list[0].shape[-2], source_X_list[0].shape[-1], args.class_num,
                          pool_mode=args.pool, f1=args.f1, d=2, f2=args.f1 * 2, kernel_length=64,
                          drop_prob=args.dropout, source_num=len(source_id_list))'''
    backbone = CEDLCross(source_X_list[0].shape[-2], source_X_list[0].shape[-1], args.class_num,len(list1))

    class_names = ['Left hand', 'Right hand', 'Foot', 'Tongue']
    print("---------------------------------configuration information-------------------------------------------------")
    for i in list(vars(args).keys()):
        print("{}:{}".format(i, vars(args)[i]))
    # -----------------------------------------------training setting--------------------------------------------------
    opt = th.optim.Adam(backbone.parameters(), lr=args.lr, weight_decay=args.w_decay)
    #cls_criterion1 = th.nn.CrossEntropyLoss()
    cls_criterion1 = Kl_loss()
    cls_criterion2 = th.nn.NLLLoss()
    if args.early_stop:
        if "high_gamma" in args.data_path:
            stop_train = EarlyStopping(patience=80, max_epochs=args.epochs)
        else:
            stop_train = EarlyStopping(patience=160, max_epochs=args.epochs)
    # -----------------------------------------------resume setting--------------------------------------------------
    best_acc = 0
    # -------------------------------------------------run------------------------------------------------------------
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.early_stop and stop_train.early_stop and epoch>500:
            print("early stop in {}!".format(epoch))
            break
        steps = train1_one_epoch(trainLoader, backbone, device, opt, cls_criterion1,
                                cls_criterion2,
                                start_time, epoch,args,source_id_list)

        avg_acc, avg_loss,all_labels,all_preds= evaluate_one_epoch1(testLoader, backbone, device, cls_criterion1, start_time,
                                               epoch, args,source_id_list)


        if args.early_stop and epoch>300:
            stop_train(avg_acc)
        save_checkpoints = {'model_classifier': backbone.state_dict(),
                            'epoch': epoch,
                            'steps': steps,
                            'acc': avg_acc}
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_epoch = epoch
            #_,_ = evaluate_best_epoch(testLoader, backbone, device, cls_criterion1, start_time,epoch, args)
            #_,_ = evaluate_other(trainLoader, backbone, device, cls_criterion1, start_time,epoch, args)
            save(save_checkpoints, os.path.join(args.model_classifier_path, 'model_best.pth.tar'))
        print('best_acc:{}  at epoch:{}'.format(best_acc, best_epoch))
        cm = confusion_matrix(all_labels, all_preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化（可选）

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Target ID: {args.target_id})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(f"_target_{args.target_id}.png")
        plt.close()
        save(save_checkpoints, os.path.join(args.model_classifier_path, 'model_newest.pth.tar')) #之后加载该模型，去掉低级分支再测试
        if 1.0 == best_acc:
            print("The modal has achieved 100% acc! Early stop at epoch:{}!".format(epoch))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str,
                        default='/home/liutaoyuan/WMB_EEGNet-master/BCICIV_2a/output',
                        help='Data path.')
    parser.add_argument('-f1', type=int, default=16,
                        help='the number of filters in EEGNet.')
    parser.add_argument('-target_id', type=int, default=2, help='Target id.')
    parser.add_argument('-dropout', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('-pool', type=str, default='mean', choices=['max', 'mean'])
    parser.add_argument('-epochs', type=int, default=800, help='Number of epochs to train.')
    parser.add_argument('-lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('-adjust_lr', type=int, default=1, choices=[0, 1, 2], help='Learning rate changes over epoch.')
    parser.add_argument('-w_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-batch_size', default=576, type=int,
                        help='batch size for training')
    parser.add_argument('-early_stop', action='store_false', help='Train early stop.')
    parser.add_argument('-print_freq', type=int, default=3, help='The frequency to show training information.')
    parser.add_argument('-father_path', type=str, default='save',
                        help='The father path of models parameters, log files.')
    parser.add_argument('-seed', type=int, default='111', help='Random seed.')
    args_ = parser.parse_args()
    train(args_)