import dgl
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, RDKFingerprint
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# GNN model definition
class GNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.residual = (in_feats == out_feats)

    def forward(self, g, h, e):
        g.ndata['h'] = h
        g.edata['e'] = e
        g.update_all(dgl.function.u_mul_e('h', 'e', 'm'), dgl.function.mean('m', 'h'))
        h = self.linear(g.ndata['h'])
        h = self.bn(h)
        if self.residual:
            h += g.ndata['h']
        return h


class GNN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, dropout=0.5, feature_size=1197):
        super(GNN, self).__init__()
        self.gnn1 = GNNLayer(in_feats, hidden_size)
        self.gnn2 = GNNLayer(hidden_size, hidden_size)
        self.gnn3 = GNNLayer(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc_feature = nn.Linear(feature_size, hidden_size)
        self.fc_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, g, features):
        h = g.ndata['h']
        e = g.edata['e']
        h = self.gnn1(g, h, e)
        h = self.gnn2(g, h, e)
        h = self.gnn3(g, h, e)
        h = self.dropout(h)
        g.ndata['h'] = h
        h_agg = dgl.mean_nodes(g, 'h')
        features_out = torch.relu(self.fc_feature(features))
        combined = torch.cat((h_agg, features_out), dim=1)
        combined = torch.relu(self.fc_combine(combined))
        return self.fc(combined)
    def get_features(self, g, features):
        h = g.ndata['h']
        e = g.edata['e']
        h = self.gnn1(g, h, e)
        h = self.gnn2(g, h, e)
        h = self.gnn3(g, h, e)
        h = self.dropout(h)
        g.ndata['h'] = h
        h_agg = dgl.mean_nodes(g, 'h')
        features_out = torch.relu(self.fc_feature(features))
        combined = torch.cat((h_agg, features_out), dim=1)
        combined = torch.relu(self.fc_combine(combined))
        return combined

class GNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GNNLayer, self).__init__()
        self.linear1 = nn.Linear(in_feats, out_feats)
        self.linear2 = nn.Linear(out_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.residual = (in_feats == out_feats)

    def forward(self, g, h, e):
        g.ndata['h'] = h
        g.edata['e'] = e
        g.update_all(dgl.function.u_mul_e('h', 'e', 'm'), dgl.function.mean('m', 'h'))
        h = torch.relu(self.linear1(g.ndata['h']))
        h = torch.relu(self.linear2(h))
        h = self.bn(h)
        if self.residual:
            h += g.ndata['h']
        return h
