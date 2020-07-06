#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *

import numpy as np

import sys
sys.path.append('../tools')
import parse, py_op
args = parse.args
kg_embedding_size = 500
kg_embedding_size = 512
use_xt_as_output = args.xt_output

def _cuda(tensor):
    if args.gpu:
        return tensor.cuda(async=True)
    else:
        return tensor

class LSTMCore(nn.Module):
    def __init__(self, opt, input_encoding_size):
        super(LSTMCore, self).__init__()
        self.input_encoding_size = input_encoding_size
        print('input_encoding_size', input_encoding_size) 
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        
        # Build a LSTM
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        # visit attention
        self.h2att =  nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, xt, state):
        # print(xt.size()) 
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = torch.max(\
            all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size),
            all_input_sums.narrow(1, 4 * self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        
        output = next_h

        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))

        output = F.relu(output)

        return output, state, None

class FCModel(nn.Module):
    def __init__(self, opt):
        super(FCModel, self).__init__()
        self.opt = opt
        self.vocab = opt.vocab
        self.vocab_size = len(opt.vocab)
        self.vocab_index = { w:i for i,w in enumerate(self.vocab) }
        self.input_encoding_size = opt.embed_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length

        self.ss_prob = 0.0 # Schedule sampling probability

        self.core = LSTMCore(opt, opt.embed_size)
        self.embed = nn.Embedding(self.vocab_size , self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, 1, False)
        self.maxpooling = nn.AdaptiveMaxPool1d(1, True)
        self.sigmoid = nn.Sigmoid()

        # infomation for knowledge graph
        self.relation = opt.relation
        self.id_icd9_dict = py_op.myreadjson(os.path.join(args.file_dir, 'id_icd9_dict.json'))
        self.cui_icd9_dict = py_op.myreadjson(os.path.join(args.file_dir, 'cui_icd9_dict.json'))
        self.icd9_cui_dict = py_op.myreadjson(os.path.join(args.file_dir, 'icd9_cui_dict.json'))

        # knowledge graph attention
        self.h2att = nn.Linear(opt.embed_size, kg_embedding_size)
        self.i2att = nn.Linear(kg_embedding_size, kg_embedding_size)
        self.att_hid_size = opt.embed_size
        self.alpha_net_kg = nn.Linear(kg_embedding_size, 1)

        # visit attention
        self.o2att = nn.Linear(opt.embed_size, opt.embed_size)
        self.alpha_net_vs = nn.Linear(opt.embed_size, 1)

        self.kg2kg = nn.Linear(kg_embedding_size, kg_embedding_size)



        # graph embedding 内容
        self.entity_id = dict()
        self.id_entity = []
        for line in open(os.path.join(args.file_dir, 'entity2id.txt')):
            data = line.strip().split()
            if len(data) == 2:
                cui, id = data[0], int(data[1])
                self.entity_id[cui] = id
                self.id_entity.append(cui)

        self.relation_id = dict()
        self.id_relation = []
        for line in open(os.path.join(args.file_dir, 'relation2id.txt')):
            data = line.strip().split()
            if len(data) == 2:
                cui, id = data[0], int(data[1])
                self.relation_id[cui] = id
                self.id_relation.append(cui)

        self.edge_dict = { }
        for line in open(os.path.join(args.file_dir, 'graph.txt')):
            data = line.strip().split('\t')
            node_f,node_s,relation_type = int(data[0]), int(data[1]), int(data[2])
            # if relation_type == 0:
            #     self.edge_dict[node_f] = self.edge_dict.get(node_f, []) + [[node_s, int(relation_type)]]
            self.edge_dict[node_f] = self.edge_dict.get(node_f, []) + [[node_s, int(relation_type)]]

        if args.model == 'GRAM':
            embedding = py_op.myreadjson(os.path.join(args.file_dir, 'embedding.vec.json'))
            self.ent_embeddings = Variable(_cuda(torch.from_numpy(np.array(embedding['ent_embeddings'], dtype=np.float32))))
            self.rel_embeddings = Variable(_cuda(torch.from_numpy(np.array(embedding['rel_embeddings'], dtype=np.float32))))

        elif kg_embedding_size in [100, 200, 300]:
            embedding = py_op.myreadjson(os.path.join(args.file_dir, 'embedding.vec.json'))
            # self.ent_embeddings = Variable(torch.from_numpy(np.array(embedding['ent_embeddings'], dtype=np.float32)).cuda(async=True))
            # self.rel_embeddings = Variable(torch.from_numpy(np.array(embedding['rel_embeddings'], dtype=np.float32)).cuda(async=True))
            self.ent_embeddings = np.array(embedding['ent_embeddings'], dtype=np.float32)
            self.rel_embeddings = np.array(embedding['rel_embeddings'], dtype=np.float32)
        elif kg_embedding_size in [500, 1000]:
            self.ent_embeddings = np.zeros((len(self.entity_id), kg_embedding_size), dtype=np.float32)
            num = 0
            # wf = open('../data/kg/cui2vec_selected.csv', 'w')
            # for line in open('../data/kg/cui2vec_pretrained.csv'):
            for line in open(os.path.join(args.file_dir, 'cui2vec_selected.csv')):
                data = line.strip().split(',')
                cui = data[0].strip('"')
                vec = data[1:]
                try:
                    vec = [float(v) for v in vec]
                except:
                    continue
                if cui in self.entity_id:
                    id = self.entity_id[cui]
                    self.ent_embeddings[id] = vec
                    num += 1
                    # wf.write(line)
                # else:
                    # print(cui) 

        self.ensemble = nn.Linear(2, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.weight.data.uniform_(-initrange, initrange)
        self.ensemble.weight.data.uniform_(-initrange, initrange)
        self.ensemble.bias.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                    Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_())

    def get_relation(self, index, time): 
        relation = []
        if index > 0:
            ehr_id = str(self.vocab[index])
            if ehr_id[0] == 'C' and ehr_id in self.entity_id:
                    cui = ehr_id
                    cui = self.entity_id[cui]
                    if cui in self.edge_dict:
                        edges = self.edge_dict[cui]
                        for edge in edges:
                            node_s, relation_type = edge
                            relation.append([cui, node_s, relation_type])
                            # print(edge) 
                            # print(node_s, relation_type) 
            # '''
            elif ehr_id in self.id_icd9_dict:
                icd9 = self.id_icd9_dict[ehr_id]
                if icd9 in self.icd9_cui_dict:
                    cui = self.icd9_cui_dict[icd9]
                    if cui in self.entity_id:
                        cui = self.entity_id[cui]
                        if cui in self.edge_dict:
                            edges = self.edge_dict[cui]
                            for edge in edges:
                                # print(edge) 
                                node_s, relation_type = edge
                                relation.append([cui, node_s, relation_type])
            elif ehr_id in self.icd9_cui_dict:
                icd9 = ehr_id
                cui = self.icd9_cui_dict[icd9]
                if cui in self.entity_id:
                    cui = self.entity_id[cui]
                    if cui in self.edge_dict:
                        edges = self.edge_dict[cui]
                        for edge in edges:
                            # print(edge) 
                            node_s, relation_type = edge
                            relation.append([cui, node_s, relation_type])
            # '''

        # print(relation) 
        if len(relation):
            relation = np.array(relation, dtype=np.int64)

            # use twice attention
            if 0 and time > 0:
                new_relation = [relation]
                for rel in relation:
                    # print(rel) 
                    _, node, edge = rel
                    rel2 = self.get_relation(node, time - 1)
                    if len(rel2):
                        new_relation.append(rel2)
                new_relation = np.concatenate(new_relation)
                relation = new_relation
            # print(relation) 

            assert len(relation.shape) == 2


        return relation

    def kg_attention(self, att_h, relation):


        # 以下是全局相同的 embedding
        use_attention = 1
        if use_attention:
            # attention
            target_ids = [l[1] for l in relation]
            cuis = [self.id_entity[id] for id in target_ids]
            vocab_ids = [self.vocab_index[cui] for cui in cuis]
            vocab_ids = Variable(_cuda(torch.from_numpy(np.array(vocab_ids, dtype=np.int64))))
            embeddings = self.embed(vocab_ids)

            # expand
            att_h = att_h.expand_as(embeddings)
            att = self.i2att(embeddings)

            # compute weight
            dot = att + att_h                                       # att_size * att_hid_size
            dot = F.tanh(dot)                                       # att_size * att_hid_size
            dot = self.alpha_net_kg(dot)                               # (batch * att_size) * 1

            # soft attention
            dot = dot.view(1, -1)
            weight = F.softmax(dot)
            att_res = torch.mm(weight, embeddings).squeeze()

            return att_res, weight.data.cpu().numpy()
        else:
            # 直接max pooling， 在50-100长度时有效
            target_ids = [l[1] for l in relation]
            cuis = [self.id_entity[id] for id in target_ids]
            vocab_ids = [self.vocab_index[cui] for cui in cuis]
            vocab_ids = Variable(_cuda(torch.from_numpy(np.array(vocab_ids, dtype=np.int64))))
            embeddings = self.embed(vocab_ids)
            embeddings = torch.transpose(embeddings, 1, 0)
            embeddings = embeddings.view((1, embeddings.size(0), embeddings.size(1)))
            embeddings = F.adaptive_max_pool1d(embeddings, 1)
            pool_res = embeddings.view(-1)
        
            return pool_res



        # 以下是另外embedding 的 attention
        '''
        # relation = Variable(torch.from_numpy(relation).cuda(async=True))
        source_node = relation[:,0]
        target_node = relation[:,1]
        relation = relation[:,2]
        source_embedding = self.ent_embeddings[source_node]
        target_embedding = self.ent_embeddings[target_node]
        if kg_embedding_size in [100, 200, 300]:
            relation_embedding = self.rel_embeddings[relation]
        if kg_embedding_size == 100:
            edge_embedding = target_embedding
        elif kg_embedding_size == 200:
            edge_embedding = np.concatenate((relation_embedding, target_embedding), 1)
        elif kg_embedding_size == 300:
            edge_embedding = np.concatenate((source_embedding, relation_embedding, target_embedding), 1)
        elif kg_embedding_size == 500:
            edge_embedding = target_embedding
        elif kg_embedding_size == 1000:
            edge_embedding = np.concatenate((source_embedding, target_embedding), 1)
        


        node_embedding = Variable(torch.from_numpy(edge_embedding).cuda(async=True))
        att_h = att_h.expand_as(node_embedding)
        att = self.i2att(node_embedding)


        dot = att + att_h                                       # att_size * att_hid_size
        dot = F.tanh(dot)                                       # att_size * att_hid_size
        dot = self.alpha_net_kg(dot)                               # (batch * att_size) * 1

        # soft attention
        dot = dot.view(1, -1)
        weight = F.softmax(dot)
        att_res = torch.mm(weight, node_embedding).squeeze()

        # hard attention
        # dot = dot.view(-1)                                   # att_size
        # _, idcs = torch.topk(dot, 1)
        # att_res = torch.index_select(node_embedding, 0, idcs).view(kg_embedding_size)
        
        return att_res
        '''

    def forward(self, seq, mask, label=None):


        # nll = Variable(torch.from_numpy(np.zeros((1), dtype=np.int64)).cuda(async=True))
        # nll_embedding = Variable(torch.from_numpy(np.zeros((kg_embedding_size), dtype=np.float32)).cuda(async=True))
        nll_embedding = Variable(_cuda(torch.from_numpy(np.zeros((512), dtype=np.float32))))

        batch_size = seq.size(0)
        state = self.init_hidden(batch_size)
        state_kg = self.init_hidden(batch_size)
        outputs = []
        outputs_kg = []

        graph_att_res =[]

        for i in range(seq.size(1)):


            it = seq[:, i-1].clone()
            xt = self.embed(it)

            output, state, att_h = self.core(xt, state)
            outputs.append(output)

            # infomation for knowledge graph
            tail_embedding = []
            data_seq = seq[:, i-1].data.cpu().numpy()

            # 这里可以用并发加速
            att_res = []
            graph_att_res_step = []
            for j, index in enumerate(data_seq):
                att_w = []
                relation = self.get_relation(index, self.opt.time)
                if len(relation):
                    att_h = self.h2att(state[1])
                    # att_h = self.h2att(state_kg[0])
                    res, att_w = self.kg_attention(att_h[0][j], relation)
                    att_res.append(res)
                else:
                    att_res.append(nll_embedding)
                if label is not None:
                    graph_att_res_step.append([relation, att_w])
            att_res = torch.stack(att_res)

            if self.opt.use_cat:
                xt = torch.cat((xt, att_res), 1)
            else:
                xt = att_res
            output, state, att_h = self.core(xt, state)
            outputs.append(output)

            if label is not None:
                graph_att_res.append(graph_att_res_step)

        # global pooling
        outputs = torch.stack(outputs, 2)
        pool_output, indices  = self.maxpooling(outputs)

        pool_output = torch.squeeze(pool_output)
        if self.opt.use_gp:
            output = pool_output


        out1 = self.logit(output)
        out1 = self.sigmoid(out1)


        # calculate comtributions
        if label is not None:

            pool_output = pool_output.data.cpu().numpy()
            indices = indices.data.cpu().numpy()
            if indices.shape[0] == 2:
                indices = indices[1]
            indices = np.reshape(indices, [batch_size, 512])
            assert pool_output.shape == indices.shape
            logit_weight = self.logit.weight.data.cpu().numpy().reshape(-1)
            contributions = np.zeros((batch_size, outputs.size(2)))
            for i in range(pool_output.shape[0]):
                for j in range(pool_output.shape[1]):
                    con = pool_output[i,j] * logit_weight[j]
                    idx = indices[i, j]
                    contributions[i,idx] += con
            outputs = outputs.data.cpu().numpy()
            return out1, contributions, outputs, logit_weight, graph_att_res
        else:
            return out1



def hard_mining(neg_output, neg_labels, num_hard, largest=True):
    num_hard = min(max(num_hard, 10), len(neg_output))
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)), largest=largest)
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.classify_loss = nn.BCELoss()

    def forward(self, prob, labels, train=True):

        pos_ind = labels > 0.5
        neg_ind = labels < 0.5
        pos_label = labels[pos_ind]
        neg_label = labels[neg_ind]
        pos_prob = prob[pos_ind]
        neg_prob = prob[neg_ind]
        pos_loss, neg_loss = 0, 0

        # hard mining
        num_hard_pos = 2
        num_hard_neg = 6
        if args.hard_mining:
            pos_prob, pos_label= hard_mining(pos_prob, pos_label, num_hard_pos, largest=False)
            neg_prob, neg_label= hard_mining(neg_prob, neg_label, num_hard_neg, largest=True)

        if len(pos_prob):
            pos_loss = 0.5 * self.classify_loss(pos_prob, pos_label) 

        if len(neg_prob):
            neg_loss = 0.5 * self.classify_loss(neg_prob, neg_label)
        classify_loss = pos_loss + neg_loss
        # classify_loss = self.classify_loss(prob, labels)

        # stati number
        prob = prob.data.cpu().numpy() > 0.5
        labels = labels.data.cpu().numpy()
        pos_l = (labels==1).sum()
        neg_l = (labels==0).sum()
        pos_p = (prob + labels == 2).sum()
        neg_p = (prob + labels == 0).sum()

        return [classify_loss, pos_p, pos_l, neg_p, neg_l]

