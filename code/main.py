#!/usr/bin/env python
# coding=utf-8

__author__ = "Changchang.Yin"


import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
import sys
import time
import numpy as np
from sklearn import metrics
import random
import json
from collections import OrderedDict
from tqdm import tqdm


import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader

import data_loader
import model

sys.path.append('../tools')
import parse, py_op

args = parse.args
if args.use_xt:
    args.use_kg = 0
    args.use_gp = 1
    args.lr = 0.001
if args.xt_output:
    args.lr = 0.001
if args.model == 'GRAM':
    args.use_kg = 0
    args.embed_size = 100
    args.lr = 0.001
if args.model == 'KEMA':
    args.use_cat = 1
    args.use_kg = 1
if torch.cuda.is_available():
    args.gpu = 1
else:
    args.gpu = 0
print 'epochs,', args.epochs

def _cuda(tensor, is_tensor=True):
    if args.gpu:
        if is_tensor:
            return tensor.cuda(async=True)
        else:
            return tensor.cuda()
    else:
        return tensor

def get_lr(epoch):
    lr = args.lr
    return lr

    if epoch <= args.epochs * 0.5:
        lr = args.lr
    elif epoch <= args.epochs * 0.75:
        lr = 0.1 * args.lr
    elif epoch <= args.epochs * 0.9:
        lr = 0.01 * args.lr
    else:
        lr = 0.001 * args.lr
    return lr

# def compute_auc()
def print_contributions(contributions):
    for i in range(10):
            for j in range(200,250, 2):
                x = int(contributions[i,j]*100,)
                if x > 99:
                    x = 99
                if x < -9:
                    x = -9
                x = str(x)
                while(len(x) < 3):
                    x = ' '+x
                print x,
            print

def analyse_contributions(labels, contributions, raw_data):
    cui_con_dict = { }
    # py_op.myreadjson('../result/cui_con_dict.json')

    print labels.shape, contributions.shape, raw_data.shape
    # print err

    pos_labels = labels[labels>0.5]
    pos_contributions = contributions[labels>0.5, :]
    pos_contributions[pos_contributions < 0] = 0
    pos_data = raw_data[labels>0.5, :]
    # print pos_labels.shape, pos_contributions.shape
    # print err

    pos_sum = (pos_contributions>0) * pos_contributions + 0.0001
    pos_sum = np.fabs(pos_sum.sum(1)).reshape((-1,1))
    # pos_sum = np.fabs(pos_contributions.sum(1)).reshape((-1,1))

    pos_ratio = pos_contributions / pos_sum
    # print_contributions(pos_contributions)

    # 不能reshape
    # pos_ratio = pos_ratio.reshape(-1)
    # pos_data = pos_data.reshape(-1)

    id_name_dict = py_op.myreadjson(os.path.join(args.file_dir, 'id_name_dict.json'))
    ehr_cui_dict = py_op.myreadjson(os.path.join(args.file_dir, 'ehr_cui_dict.json'))
    cui_distance_dict = py_op.myreadjson(os.path.join(args.file_dir, 'cui_distance_dict.json'))
    name_id_dict = { v:ehr_cui_dict.get(k, k) for k,v in id_name_dict.items() }
    name_ratio_dict = { }
    for b_ration, b_idx in zip(pos_ratio, pos_data):
        idx_ratio = dict()
        for ratio, idx in zip(b_ration, b_idx):
            idx_ratio[idx] = idx_ratio.get(idx, 0) + ratio

        for idx, ratio in idx_ratio.items():
            id = args.vocab[idx]
            id = str(id)
            if idx>0 and id in id_name_dict:
                name = id_name_dict[id]
                name_ratio_dict[name] = name_ratio_dict.get(name, []) + [ratio]
                cui_con_dict[id] = cui_con_dict.get(id, []) + [ratio]

    cons_dir = '../result/cons'
    num = len(os.listdir(cons_dir))
    # py_op.mywritejson('../result/cons/{:d}.json'.format(num), cui_con_dict)
    contributions_list = []

    name_score_dict = { }
    for n,v in name_ratio_dict.items():
        if len(v) > 4:
            name_score_dict[n] = np.mean(v)
    name_list = sorted(name_score_dict.keys(), key=lambda n:- name_score_dict[n])
    for name in name_list[:30]:
        # if name_id_dict[name] in cui_distance_dict:
        #     print 'contribution rate: {:3.2f}%  {:d}    {:s}    {:s}'.format(100 * name_score_dict[name], cui_distance_dict.get(name_id_dict[name], -1), name_id_dict[name], name)
        # print name_ratio_dict[name]
        print 'contribution rate: {:3.4f}%  {:d}    {:s}    {:s}'.format(100 * name_score_dict[name], cui_distance_dict.get(name_id_dict[name], -1), name_id_dict[name], name)
        # contributions_list.append({ 'id': name_id_dict[name], 'contribution': name_score_dict[name] })
    print '一共{:d}个name'.format(len(name_list))
    return cui_con_dict






def test(data_loader, net, loss, epoch, best_auc, phase='valid', cui_con_dict=[]):
    net.eval()
    results = np.zeros(4) + 0.00001
    loss_list = []
    outputs = []
    labels = []
    contributions = []
    raw_data = []
    kg_data = []
    for b, (data, label, mask) in enumerate(tqdm(data_loader)):
        data = Variable(_cuda(data)) # [bs, 250]
        label = Variable(_cuda(label)) # [bs, 1]
        mask = Variable(_cuda(mask)) # [bs, 1]
        # output, con = net(data, mask, label)
        net_out = net(data, mask, label)
        output, con, output_vectors, fc_weight = net_out[:4]
        if args.phase == 'test':
            for i_o in range(len(output_vectors)):
                i_p = b * args.batch_size + i_o
                save_dir = '/'.join(args.resume.split('/')[:-1])
                np.save(os.path.join(save_dir, str(i_p) + '_h.npy'), output_vectors[i_o])
                fc_file = os.path.join(save_dir, 'fc_weight.npy')
                if not os.path.exists(fc_file):
                    np.save(fc_file, fc_weight)
                if i_p == 0:
                    print output
        contributions.append(con)
        loss_output = loss(output, label)
        outputs = outputs + list(output.data.cpu().numpy().reshape([-1]))
        labels = labels + list(label.data.cpu().numpy().reshape([-1]))

        # raw_data.append(data.data.cpu().numpy())
        indices_data = data.data.cpu().numpy()
        if args.use_kg:
            if args.use_cat == 0:
                new_indices = []
                kg_indices = []
                for inds in indices_data:
                    new_inds = []
                    new_kgs = []
                    for ind in inds:
                        new_inds.append(ind)
                        new_inds.append(ind)
                        new_kgs.append(1)
                        new_kgs.append(0)
                    new_indices.append(new_inds)
                    kg_indices.append(new_kgs)
                # print indices_data[0]
                indices_data = np.array(new_indices)
                kg_indices = np.array(kg_indices)
                # print indices_data[0]
                kg_data.append(kg_indices)
        raw_data.append(indices_data)
        


        for i in range(4):
            results[i] += loss_output[i+1]
        loss_list.append(loss_output[0].data.cpu().numpy())


    labels = np.array(labels)
    raw_data = np.concatenate(raw_data)
    if len(kg_data): 
        kg_data = np.concatenate(kg_data)
    # print raw_data.shape
    # print_contributions(contributions)

    outputs = np.array(outputs)
    fpr, tpr, threshholds = metrics.roc_curve(labels, outputs)
    auc = metrics.auc(fpr, tpr)


    print('%s Epoch %03d ' % (phase, epoch))
    print 'loss: {:3.4f}'.format(np.mean(loss_list))
    print 'tpr: {:3.4f}\t tnr: {:3.4f}\t acc: {:3.4f}\t auc: {:3.4f}'.format(results[0]/results[1], results[2]/results[3], (results[0] + results[2])/(results[1] + results[3]), auc), 
    print('\t tp: %04d/%04d\t tn: %04d/%04d' % (results[0], results[1], results[2], results[3]))


    if auc > best_auc[0]:
        best_auc[0:2] = auc, epoch
        if args.phase == 'train':
            torch.save(net.state_dict(), os.path.join(args.result_dir, 'models', 'kg-{:d}-gp-{:d}-auc-{:3.2f}'.format(args.use_kg, args.use_gp, auc)))
        # if args.use_kg == 2:
        '''
        contributions = np.concatenate(contributions)
        cui_con = analyse_contributions(labels, contributions, raw_data)
        if len(kg_data):
            kg_con = analyse_contributions(labels, contributions, kg_data)
            cui_con_dict = { 'cui': cui_con, 'kg': kg_con.values()[0] }
        else:
            cui_con_dict = cui_con
        '''
    # if args.use_xt and auc > 0.9:
    #     contributions = np.concatenate(contributions)
    #     analyse_contributions(labels, contributions, raw_data)
    print 'best auc\n\t epoch {:d} auc {:3.4f} \t'.format(best_auc[1], best_auc[0])
    print
    print
    return best_auc, cui_con_dict

def train(data_loader, net, loss, epoch, optimizer, best_auc):

    net.train()
    # optimizer_all, optimizer_kg, optimizer_ehr = optimizer 
    # optimizer = optimizer[0]
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    results = np.zeros(4) + 0.00001
    loss_list = []
    loss_list_kg = []
    for b, (data, label, mask) in enumerate(tqdm(data_loader)):
        # if b > 10:
        #     break
        data = Variable(_cuda(data)) # [bs, 250]
        label = Variable(_cuda(label)) # [bs, 1]
        mask = Variable(_cuda(mask)) # [bs, 1]
        output = net(data, mask) # [bs, 1]
        loss_output = loss(output, label)



        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        for i in range(4):
            results[i] += loss_output[i+1]
        loss_list.append(loss_output[0].data.cpu().numpy())

    print('Train Epoch %03d (lr %.5f)' % (epoch, lr))
    # print 'loss: {:3.4f} {:3.4f} {:3.4f}'.format(np.mean(loss_list))
    print 'loss: {:3.4f} \t'.format(np.mean(loss_list))
    print 'init    tpr: {:3.4f}\t tnr: {:3.4f} '.format(results[0]/results[1], results[2]/results[3])

def main():
    dataset = data_loader.DataBowl(args, phase='train')
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    dataset = data_loader.DataBowl(args, phase='valid')
    valid_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    args.vocab = dataset.vocab
    args.relation = dataset.relation

    # net, loss = model.Net(args), model.Loss()
    net, loss = model.FCModel(args), model.Loss()

    net = _cuda(net, 0)
    loss = _cuda(loss, 0)

    parameters_all = []
    for p in net.parameters():
        parameters_all.append(p)

    optimizer = torch.optim.Adam(parameters_all, args.lr)

    best_auc = [0,0,0,0,0,0]

    cui_con_dict = { }
    if args.phase == 'train':
        for epoch in range(args.epochs):
            train(train_loader, net, loss, epoch, optimizer, best_auc)
            best_auc, cui_con_dict= test(valid_loader, net, loss, epoch, best_auc, 'valid', cui_con_dict)
            print args.words



        if 1:
            cons_dir = '../result/cons/{:s}/{:d}'.format(args.model, args.predict_day)
            py_op.mkdir(cons_dir)
            num = len(os.listdir(cons_dir))
            py_op.mywritejson(os.path.join(cons_dir,'{:d}.json'.format(num)), cui_con_dict)
            # break

        print 'best auc', best_auc
        auc = best_auc[0]
        with open('../result/log.txt', 'a') as f:
            f.write('#model {:s} #auc {:3.4f}\n'.format(args.model, auc))


    elif args.phase == 'test':
        net.load_state_dict(torch.load(args.resume))
        test(valid_loader, net, loss, 0, best_auc, 'valid', cui_con_dict)

if __name__ == '__main__':
    main()
