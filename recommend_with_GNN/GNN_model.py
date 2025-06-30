#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# from neomodel import db, config

# from global_settings import *

import torch

import torch.nn as nn

from torch_geometric.nn import GCNConv

from torch_geometric.datasets import KarateClub
from torch_geometric.data import DataLoader

import torch.nn.functional as F

import numpy as np


class Route_GNN(nn.Module):
    def __init__(self, num_features, hidden_features, num_classes):
        super(Route_GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.conv3 = GCNConv(hidden_features, 8)
        self.classifier = nn.Linear(8, num_classes)
        
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.conv3(h, edge_index)
        h = F.relu(h)
        # out = F.log_softmax(h, dim=1)
        out = self.classifier(h)
        return out


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for iter, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # print(f'test:{out[data.train_mask]},true:{data.y[data.train_mask]}')
        loss = criterion(out, data.y)
        loss.backward()
        # loss_list.append(loss.item())
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter+1)
    return epoch_loss

def test(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total_node = 0
    epoch_loss = 0
    for iter, data in enumerate(test_loader):
        data = data.to(device)
        out = model(data)
        # print(out)
        loss = criterion(out, data.y)
        pred = out.max(dim=1)[1]
        # print(pred)
        correct += pred.eq(data.y).sum().item()
        total_node += data.num_nodes
        epoch_loss += loss.detach().item()
    correct_rate = correct/total_node
    epoch_loss /= (iter+1)
    return correct_rate, epoch_loss

def test_with_data(model, test_loader, criterion):
    correct = 0
    total_node = 0
    epoch_loss = 0
    
    data_y_list = []
    pred_y_list = []
    
    for iter, data in enumerate(test_loader):
        out = model(data)
        # print(out)
        loss = criterion(out, data.y)
        pred = out.max(dim=1)[1]
        # print(pred)
        correct += pred.eq(data.y).sum().item()
        total_node += data.num_nodes
        epoch_loss += loss.detach().item()
        
        data_y_list.append(data.y.numpy())
        pred_y_list.append(pred.numpy())
    
    data_y_list = np.concatenate(data_y_list, axis=0)
    pred_y_list = np.concatenate(pred_y_list, axis=0)
        
    correct_rate = correct/total_node
    epoch_loss /= (iter+1)
    return correct_rate, epoch_loss, data_y_list, pred_y_list

# dataset = KarateClub()
# data = dataset[0]


# loader = DataLoader(dataset, batch_size = 1, shuffle=True)
# print(data.y.numpy())
# print(data.train_mask)
# print(data)        
        
# model = Route_GNN(34, 64, 4)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(model)
# model = model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# loss_list = []


# for epoch in range(501):
#     loss = train(model, loader, optimizer, device)
#     acc = test(model, loader, device)
#     if epoch % 50 == 0:
#         print(f'Epoch:{epoch}, Loss: {loss.item():.4f}, acc:{acc:.4f}')
        