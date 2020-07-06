#!/usr/bin/env python
# coding=utf-8

__author__ = "Changchang.Yin"


import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

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
from flask import Flask, jsonify


sys.path.append('../tools')
sys.path.append('models/tools')
import parse, py_op

sys.path.append('models/code')
import model
import data_loader


args = parse.args
if torch.cuda.is_available():
    args.gpu = 1
else:
    args.gpu = 0

def second_to_date(second):
    return time.localtime(second)

def _cuda(tensor):
    if args.gpu:
        return tensor.cuda(async=True)
    else:
        return tensor

def get_model(model_file=os.path.join(args.result_dir, 'mimic-kg-gp.ckpt')):
    dataset = data_loader.DataBowl(args, phase='valid')
    args.vocab = dataset.vocab
    args.relation = dataset.relation

    net, _ = model.FCModel(args), model.Loss()
    net = _cuda(net)
    net.load_state_dict(torch.load(model_file))
    return net

def get_data():
    vocab_list = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, args.dataset[:-4].lower() + 'vocab.json'))
    aid_year_dict = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'aid_year_dict.json'))
    pid_aid_did_dict = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'pid_aid_did_dict.json'))
    pid_demo_dict = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'pid_demo_dict.json'))
    case_control_data = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'case_control_data.json'))
    case_test = list(set(case_control_data['case_test'] + case_control_data['case_valid']))
    case_control_dict = case_control_data['case_control_dict']
    dataset = data_loader.DataBowl(args, phase='DGVis')
    id_name_dict = py_op.myreadjson(os.path.join(args.file_dir, 'id_name_dict.json'))
    graph_dict = { 'edge': { }, 'node': { } }
    for line in open(os.path.join(args.file_dir, 'relation2id.txt')):
        data = line.strip().split()
        if len(data) == 2:
            relation, id = data[0], int(data[1])
            graph_dict['edge'][id] = relation
    for line in open(os.path.join(args.file_dir, 'entity2id.txt')):
        data = line.strip().split()
        if len(data) == 2:
            cui, id = data[0], int(data[1])
            if cui in id_name_dict:
                graph_dict['node'][id] = id_name_dict[cui]
            else:
                graph_dict['node'][id] = cui

    aid_second_dict = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'aid_second_dict.json'))

    for pid, aid_did_dict in pid_aid_did_dict.items():
        n = 0 
        aids = sorted(aid_did_dict.keys(), key=lambda aid:int(aid), reverse=True)
        for ia, aid in enumerate(aids):
            n += len(aid_did_dict[aid])
            if n > 120:
                pid_aid_did_dict[pid] = { aid: aid_did_dict[aid] for aid in aids[:ia] }
                break



    new_pid_demo_dict = dict()
    pid_list = case_test + [c for case in case_test for c in case_control_dict[str(case)]]
    pid_list = [str(pid) for pid in pid_list]
    for pid in pid_list:
        pid = str(pid)
        demo = pid_demo_dict[pid]
        gender = demo[0]
        yob = int(demo[2:])
        if pid not in pid_aid_did_dict:
            continue
        aids = pid_aid_did_dict[pid].keys()
        year = max([aid_year_dict[aid] for aid in aids])
        age = year - yob
        assert age < 100 and age > 0
        new_pid_demo_dict[pid] = [gender, age]


    # return data
    # case_control_dict = { case: [c for c in case_control_dict[case] if c in new_pid_demo_dict] for case in case_test if case in new_pid_demo_dict}


    pid_demo_dict = new_pid_demo_dict
    pid_aid_did_dict = { pid: pid_aid_did_dict[pid] for pid in new_pid_demo_dict }

    # print('case_set', case_control_dict.keys())

    return pid_demo_dict, pid_aid_did_dict, aid_second_dict, dataset, set(case_control_dict), vocab_list, graph_dict, id_name_dict


def get_pids_data(pids, pid_aid_did_dict, aid_second_dict, dataset, case_set):
    def _tensor(data):
        data = np.array(data)
        data = torch.from_numpy(data)
        data = Variable(_cuda(data))
        return data
    data_list, mask_list, label_list, visit_list  = [], [], [], []
    for pid in pids:
        aid_did_dict = pid_aid_did_dict[pid]
        aids = sorted(aid_did_dict.keys(), key=lambda aid: aid_second_dict[aid])
        max_date = aid_second_dict[aids[-1]]
        date_did_dict = dict()
        if pid in case_set:
            is_case = 1
        else:
            is_case = 0
        for aid in aids:
            delta_date = int((aid_second_dict[aid] - max_date) / 3600 / 24)
            date_did_dict[delta_date] = aid_did_dict[aid]
        data, label, mask, n_of_visit = dataset.__getitem__([date_did_dict, is_case])
        data_list.append(data.numpy())
        mask_list.append(mask.numpy())
        label_list.append(label.numpy())
        visit_list.append(n_of_visit.numpy())
    data_list = _tensor(data_list)
    mask_list = _tensor(mask_list)
    label_list = _tensor(label_list)
    visit_list = np.array(visit_list)
    return data_list, mask_list, label_list, visit_list

def get_att(relation_att, graph_dict):
    relation, weight = relation_att
    weight = list(np.array(weight).reshape(-1))
    assert len(weight) == len(relation)

    src_edge_tgt_w = []
    for rel, w in zip(relation, weight):
        src = graph_dict['node'][rel[0]]
        edge = graph_dict['edge'][rel[2]]
        tgt = graph_dict['node'][rel[1]]
        src_edge_tgt_w.append([src, edge, tgt, w])

    # print(src_edge_tgt_w[0])

    return src_edge_tgt_w

def inference(net, data, mask, label, n_of_visit, vocab_list, pids, pid_aid_did_dict, graph_dict):
    output, contributions, output_vectors, fc_weight, graph_att_res = net(data, mask, label)

    data = data.data.cpu().numpy()
    output_vectors = output_vectors
    output = output.data.cpu().numpy()

    label = label.data.cpu().numpy().reshape(-1)
    pred = (output > 0.5).reshape(-1)
    print(sum(pred == label), len(pred))

    pid_aid_did_cr_dict = { pid: { } for pid in pids }
    pid_aid_risk_dict = { pid: { } for pid in pids }
    pid_aid_did_att_dict = { pid: { } for pid in pids }
    for ipid, pid in enumerate(pids):
        indices = data[ipid]
        crs = contributions[ipid]
        visits = n_of_visit[ipid]
        vectors = output_vectors[ipid]
        aids = sorted(pid_aid_did_dict[pid].keys(), key = lambda k:int(k))

        pid_graph_att_res = [gar[ipid] for gar in graph_att_res]

        i_visit = 1
        for iv, n in enumerate(visits):
            index = indices[iv]
            # EHR data start when index > 0
            if index > 0:
                aid = aids[n - 1]
                # print(index)
                did = vocab_list[index - 1]
                cr = list(crs[iv*2: iv*2 + 2])

                if aid not in pid_aid_did_att_dict[pid]:
                    pid_aid_did_att_dict[pid][aid] = dict()
                if aid not in pid_aid_did_cr_dict[pid]:
                    pid_aid_did_cr_dict[pid][aid] = dict()
                pid_aid_did_cr_dict[pid][aid][did] = cr

                # calculate the contribution of a visit
                if i_visit != n:
                    i_visit = aid
                    last_aid = aids[n - 2]
                    vector = vectors[:, : iv*2 - 2].max(1) # the output vectors before current visit
                    risk = np.dot(vector, fc_weight)
                    pid_aid_risk_dict[pid][last_aid] = risk

                # attention weight
                if len(pid_graph_att_res[iv][0]):
                    pid_aid_did_att_dict[pid][aid][did] = get_att(pid_graph_att_res[iv], graph_dict)
                else:
                    pid_aid_did_att_dict[pid][aid][did] = []


        # last visit's risk
        pid_aid_risk_dict[pid][aids[-1]] = output[ipid]

        # analyze graph attentin result
        # pid_aid_did_att_dict[pid]


    return pid_aid_did_cr_dict, pid_aid_risk_dict, pid_aid_did_att_dict


class DGRNN(object):
    def __init__(self):

        # build model
        self.net = get_model()

        # prepare all the test data
        test_data = get_data()
        self.pid_demo_dict, self.pid_aid_did_dict, self.aid_second_dict, self.dataset, self.case_set, \
                self.vocab_list, self.graph_dict, self.id_name_dict = test_data
        # assert len(self.pid_demo_dict) == len(self.pid_aid_did_dict)
        self.pids = list(self.pid_demo_dict.keys())
        self.icd_name_dict = py_op.myreadjson(os.path.join(args.file_dir, 'icd_name_dict.json'))

    # def test(self):
    def get_test_data(self):
        '''
        return all the data needed for visualization:
            pid_aid_did_dict: 
                pid: patient id
                aid: admission id
                did: diagnosis id
            aid_date_dict:
                aid: admission id
                date: int, admission's time
            vocab_list: diagnosis list
        '''
        aid_date_dict = { aid: second_to_date(second) for aid, second in self.aid_second_dict.items() }
        vocab_list = []
        for vocab in self.vocab_list:
            if vocab in self.id_name_dict:
                vocab_list.append(self.id_name_dict[vocab] )
            else:
                if vocab in self.icd_name_dict:
                    vocab_list.append(self.icd_name_dict[vocab])
                else:
                    vocab = vocab.strip('0')
                    if vocab in self.icd_name_dict:
                        vocab_list.append(self.icd_name_dict[vocab])
                    else:
                        vocab = vocab[:-1]
                        vocab_list.append(self.icd_name_dict[vocab.strip('0')])
                        assert len(vocab) >= 3

        return self.pid_aid_did_dict, aid_date_dict, vocab_list
        return jsonify(self.pid_aid_did_dict, aid_date_dict, vocab_list)

    # def test(self, pid_aid_did_dict = { }):
    def predict(self, pid_aid_did_dict):
        '''
        predict the risk for given patients
        intput: 
            pid_aid_did_dict
                pid: patient id
                aid: admission id
                did: diagnosis id
        output:
            pid_aid_did_cr_dict:
            pid_aid_risk_dict:
            pid_aid_did_att_dict:
        '''

        # pid_aid_did_dict = { k:v for k,v in list(self.pid_aid_did_dict.items())[:10] }

        pids_batch = pid_aid_did_dict.keys()
        data, mask, label, n_of_visit = get_pids_data(pids_batch, pid_aid_did_dict, \
                self.aid_second_dict, self.dataset, self.case_set) 
        pid_aid_did_cr_dict, pid_aid_risk_dict, pid_aid_did_att_dict = inference(self.net, data, \
                mask, label, n_of_visit, self.vocab_list, pids_batch, pid_aid_did_dict, self.graph_dict)

        for pid, aids in pid_aid_did_cr_dict.items():
            for aid, dids in aids.items():
                for did, cr in dids.items():
                    pid_aid_did_cr_dict[pid][aid][did] = [str(x) for x in cr]

        for pid, aids in pid_aid_risk_dict.items():
            for aid, risk in aids.items():
                pid_aid_risk_dict[pid][aid] = str(risk)

        for pid, aids in pid_aid_did_att_dict.items():
            for aid, dids in aids.items():
                for did, att in dids.items():
                    pid_aid_did_att_dict[pid][aid][did] = [str(x) for x in att]

        return pid_aid_did_cr_dict, pid_aid_risk_dict, pid_aid_did_att_dict
        return jsonify(pid_aid_did_cr_dict, pid_aid_risk_dict, pid_aid_did_att_dict )

    def generate_csv(self):

        print(self.case_set)

        pids = list(self.pid_aid_did_dict.keys())
        vectors = []
        for i in tqdm(range(0, len(pids), args.batch_size)):
            pids_batch = pids[i: i+args.batch_size]
            pid_aid_did_dict = { pid: self.pid_aid_did_dict[pid] for pid in pids_batch }
            data, mask, label, n_of_visit = get_pids_data(pids_batch, pid_aid_did_dict, \
                self.aid_second_dict, self.dataset, self.case_set) 
            _, _, output_vectors, _, _ = self.net(data, mask, label)
            vectors.append(output_vectors.max(2)) 
        vectors = np.concatenate(vectors, 0)
        assert len(pids) == len(vectors)

        # use tsne
        from sklearn.manifold import TSNE
        X_tsne = TSNE(n_components=2,learning_rate=100).fit_transform(vectors)

        wdemo = open(os.path.join(args.result_dir, 'demo.csv'), 'w')
        wdemo.write('PID,GENDER,AGE,X,Y,LABEL\n')
        for pid, xy in zip(pids, vectors):
            if pid in self.case_set:
                label = 1
            else:
                label = 0
            demo = self.pid_demo_dict[pid]
            wdemo.write(pid + ',')
            wdemo.write(demo[0]+ ',')
            wdemo.write(str(demo[1])+ ',')
            wdemo.write(str(xy[0])+ ',')
            wdemo.write(str(xy[1])+ ',')
            wdemo.write(str(label)+ '\n')
            




def main():
    dr = DGRNN()
    dr.generate_csv()
    return

    pids = dr.pids[:10]
    pid_aid_did_dict = { k:dr.pid_aid_did_dict[k] for k in pids }

    dr.get_test_data()
    # print(data[-1])
    dr.predict(pid_aid_did_dict)



if __name__ == '__main__':
    main()
