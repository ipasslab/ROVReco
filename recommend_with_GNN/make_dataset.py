#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from neomodel import db, config

from global_settings import *
from utils import *
from analyse_in_graph import decide_AS_range

import pandas as pd
import os
import torch
from torch_geometric.data import Data

import multiprocessing
from tqdm import tqdm

import numpy as np

BLOCK_SIZE = 10000

import time

def init_route(collect_time):
    route_csv_path = os.path.join(ORIGIN_ROUTE_SAVE_PATH, collect_time, "all_routes.csv")
    # route_csv_path = os.path.join("/home/hunter/object/RPKI_code/origin_route_path/20240701/all_routes.csv")
    route_total_df = pd.read_csv(route_csv_path)
    r_df = route_total_df.drop_duplicates()
    r_df = r_df[r_df['path_len']>=3]
    
    return r_df 
    # r_df.to_csv('process_route.csv', index=False)
    # print(r_df)
    
def get_related_route(prefix, collect_time):
    # as_connect_df = decide_AS_range.get_basic_topology_of_AS_by_prefix(prefix)
    subgraph_as_list = decide_AS_range.get_listofAS_hasrouteto_prefix(prefix)
    
    # route_csv_path = "process_route.csv"
    route_df = init_route(collect_time)
    
    # total_route_len = len(route_df)
    # multi_core = MULTI_CORE_NUM
    # d_len = total_route_len//multi_core
    # mp = multiprocessing.Pool(multi_core)
    
    # route_devided = []
    
    # for i in range(0, multi_core-1):
    #     devide_data = route_df[i*d_len:(i+1)*d_len]
    #     route_devided.append(
    #         mp.apply_async(
    #             func=multi_finder,
    #             kwds={'data_df':devide_data, 'as_list':subgraph_as_list, 'process_id':i}))
    # devide_data = route_df[d_len*(multi_core-1):total_route_len]
    # route_devided.append(
    #     mp.apply_async(
    #         func=multi_finder,
    #         kwds={'data_df':devide_data, 'as_list':subgraph_as_list, 'process_id':i}))
    
    # mp.close()
    # mp.join()
    
    df_list = []
    
    for as_num in subgraph_as_list:
        # print(as_num)
        related_df = route_df[route_df['as_path'].str.contains(as_num)]
        df_list.append(related_df)
    
    # for result in tqdm(route_devided):
    #     part_res = result.get()
    #     df_list.append(part_res)
    
    total_r_df = pd.concat(df_list)
    total_r_df = total_r_df.drop_duplicates()
    # total_r_df.to_csv("subgraph_route.csv", index=False)
    # total_r_df = pd.read_csv("subgraph_route.csv")
    
    hijack_type = "subprefix"
    dataset_df = detailed_process(subgraph_as_list, total_r_df, prefix, hijack_type)
    
    prefix_for_file = prefix.replace("/", "_")
    data_path = os.path.join(GNN_DATA_PATH, collect_time, f"{prefix_for_file}_{SERVE_COMPLETENESS_STANDARD}_all.csv")
    print(data_path)
    check_path_directory(data_path)
    dataset_df.to_csv(data_path, index=False)
    
    process_log("Related route data has been done")
    return dataset_df
 
def multi_finder(data_df, as_list, process_id):
    start_time = time.time()
    result_df = pd.DataFrame(columns=['prefix', 'as_path', 'path_len'])
    count = 0
    for index, row in data_df.iterrows():
        as_path = row['as_path']
        
        for asn in as_list:
            if asn in as_path:
                result_df.loc[count] = row
                count +=1
                break
    end_time = time.time()
    print(f"Process {process_id}: Processd {len(data_df)} data. Processing for {(end_time-start_time)/60} minutes\n")        
    
    return result_df    
    
def detailed_process(as_list, total_r_df, prefix, type):
    rov_list = get_ROV_recommend(prefix, type)
    print(f"ROV: {rov_list}")
    
    total_route_len = len(total_r_df)
    
    multi_core = MULTI_CORE_NUM
    d_len = total_route_len//multi_core
    mp = multiprocessing.Pool(multi_core)
    
    route_devided = []
    
    for i in range(0, multi_core-1):
        devide_data = total_r_df[i*d_len:(i+1)*d_len]
        route_devided.append(
            mp.apply_async(
                func=mult_process_unit,
                kwds={'data_df':devide_data, 'as_list':as_list, 'rov_list':rov_list, 'process_id':i}))
    devide_data = total_r_df[d_len*(multi_core-1):total_route_len]
    route_devided.append(
        mp.apply_async(
            func=mult_process_unit,
            kwds={'data_df':devide_data, 'as_list':as_list, 'rov_list':rov_list, 'process_id':i}))
    
    mp.close()
    mp.join()
    
    route_dataset_list = []
    
    for result in tqdm(route_devided):
        part_res = result.get()
        route_dataset_list.append(part_res)
    dataset_frame = pd.concat(route_dataset_list, ignore_index=True)
    dataset_frame = dataset_frame.drop_duplicates()
    
    return dataset_frame
    
    
def mult_process_unit(data_df, as_list, rov_list, process_id):
    start_time = time.time()
    
    count = 0
    
    total_data_list = []
    
    for index, row in data_df.iterrows():
        as_path = row['as_path']
        # abandon
        if '{' in as_path:
            continue
        
        origin_list = as_path.split(',')
        path_as_list = list(set(origin_list))
        path_as_list.sort(key=origin_list.index)
        
        cut_i = 0
        # cut_i for origin from subgraph
        for i in range(len(path_as_list)):
            as_num = path_as_list[i]
            if as_num in as_list:
                cut_i = i
                break
        cut_path = path_as_list[cut_i:]
        
        rov_sign_mask = []
        has_rov = 0
        new_path_len = len(cut_path)
        for i in range(new_path_len):
            
            as_num = cut_path[i]
            if as_num in rov_list:
                rov_sign_mask.append('1')
                has_rov += 1
            else:
                rov_sign_mask.append('0')
        count += 1
        new_as_path = ','.join(cut_path)
        rov_mask = ','.join(rov_sign_mask)
        
        total_data_list.append([new_as_path, new_path_len, rov_mask, has_rov])
        
        
        # if count % BLOCK_SIZE == 0:
        #     temp_time = time.time()
        #     print(f"Process {process_id}: Processd {count} data. Processing for {(temp_time-start_time)/60} minutes\n")
    
    column_names = ['as_path', 'path_len', 'rov_mask', 'has_rov']
    new_dataframe = pd.DataFrame(total_data_list, columns=column_names)
    
    new_dataframe = new_dataframe.drop_duplicates()
    
    end_time = time.time()
    print(f"Process {process_id}: Processd {count} data. Processing for {(end_time-start_time)/60} minutes\n")
    
    return new_dataframe
    
def get_ROV_recommend(prefix, type):
    prefix_for_file = prefix.replace("/", "_")
    if type == "subprefix":
        prefix_file_path = os.path.join(ANALYSE_PROCESS_PATH, 'subprefix_hijacking', str(SERVE_COMPLETENESS_STANDARD), prefix_for_file)
    elif type == "prefix":
        prefix_file_path = os.path.join(ANALYSE_PROCESS_PATH, 'prefix_hijacking', str(SERVE_COMPLETENESS_STANDARD), prefix_for_file)
    data_save_path = os.path.join(prefix_file_path, "ROV")
    rov_path = os.path.join(data_save_path, "ROV_iteration.csv")
    rov_df = pd.read_csv(rov_path)
    rov_list = rov_df.iloc[-1]['ROV'].split(',')
    return rov_list


def make_adj_matrix(as_path_len):
    num_nodes = as_path_len
    # init matrix 
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(as_path_len-1):
        adj_matrix[i, i+1] = 1
    adj_matrix = np.matrix(adj_matrix).T
    adj_matrix = adj_matrix + adj_matrix.T
    e_matrix = np.identity(np.array(adj_matrix).shape[0])
    adj_matrix = adj_matrix + e_matrix
    adj_matrix_nonzero = np.array(adj_matrix).nonzero()
    
    adj_matrix_nonzero = np.array(adj_matrix_nonzero)
    # print(adj_matrix_nonzero)
    # exit()
    return torch.tensor(adj_matrix_nonzero)

def make_x_feature(as_list, node_emb_df):
    feature_list = []
    for as_num in as_list:
        as_num = int(as_num)
        emb = node_emb_df.loc[as_num]['embedding']
        node_emb = [float(num) for num in emb.strip('[]').split(',')]
        feature_list.append(node_emb)

    feature_node = torch.tensor(feature_list, dtype=torch.float)
    # print(feature_node)
    # exit()
    return feature_node

def make_y_label(rov_mask):
    node_emb = [int(num) for num in rov_mask.split(',')]
    return torch.tensor(node_emb)
    
def build_graph(route_df, build_num):
    graph_data = []
    route_df_len = len(route_df)
    
    node_embedding_path = os.path.join(GDS_DATA_PATH, "as_embedding.csv")
    
    node_embedding_df = pd.read_csv(node_embedding_path)
    node_embedding_df = node_embedding_df.set_index("as_num")
    # print(node_embedding_df)
    # print(node_embedding_df.dtypes)
    
    # emb = node_embedding_df.loc[3356]['embedding']
    # embedding_list = [float(num) for num in emb.strip('[]').split(',')]
    # print(embedding_list)
    # exit()
    
    sample_df = route_df.sample(n=build_num).reset_index(drop=True)
    print(sample_df)
    
    for j in tqdm(range(0, build_num)):
        as_path_len = int(sample_df.loc[j]['path_len'])
        adj_matrix_nonzero = make_adj_matrix(as_path_len)
        
        as_list = sample_df.loc[j]['as_path'].split(',')
        x_feature = make_x_feature(as_list, node_embedding_df)
        
        rov_mask = sample_df.loc[j]['rov_mask']
        y_label = make_y_label(rov_mask)
        
        data = Data(x = x_feature, y=y_label, edge_index=adj_matrix_nonzero)
        graph_data.append(data)
        
    return graph_data

def make_GNN_dataset(collect_time, data_count, prefix):
    
    # prefix = input("Input your prefix: ")
    # while not check_prefix_format(prefix):
    #     prefix = input("Check Your Format! Input again: ")
    
    prefix_for_file = prefix.replace("/", "_")    
    data_path = os.path.join(GNN_DATA_PATH, collect_time, f"{prefix_for_file}_{SERVE_COMPLETENESS_STANDARD}_all.csv")
    # print(data_path)
    # exit()
    if os.path.exists(data_path):
        route_df = pd.read_csv(data_path)
    else:
        process_log(f"Will build dataset for {prefix}")
        route_df = get_related_route(prefix, collect_time)
    print(route_df.shape)
    route_df = route_df[(route_df['path_len']>=3) & (route_df['has_rov']!=0)]
    print(route_df.shape)

    graph_list = build_graph(route_df, data_count)
    
    print(graph_list[0:10])
    return graph_list
    
# init_route("20240701")


def make_dataset_all():
    collect_time = "20240701"
    prefix_list = ["59.71.48.0/22","168.167.71.0/24","195.54.164.0/23","81.90.208.0/20","45.160.192.0/22","144.208.192.0/22","102.68.115.0/24"]
    
    all_list = []
    for prefix in prefix_list:
        prefix_for_file = prefix.replace("/", "_")    
        data_path = os.path.join(GNN_DATA_PATH, collect_time, f"{prefix_for_file}_{SERVE_COMPLETENESS_STANDARD}_all.csv")
        # print(data_path)
        # exit()
        if os.path.exists(data_path):
            route_df = pd.read_csv(data_path)
        else:
            process_log(f"Will build dataset for {prefix}")
            route_df = get_related_route(prefix, collect_time)

        route_df = route_df[(route_df['path_len']>=3) & (route_df['has_rov']!=0)]
        print(route_df.shape)

        graph_list = build_graph(route_df, 500000)
        
        all_list.extend(graph_list)
        
    return all_list


def analyse_rov_data(collect_time, prefix):
    
    prefix_for_file = prefix.replace("/", "_")    
    data_path = os.path.join(GNN_DATA_PATH, collect_time, f"{prefix_for_file}_{SERVE_COMPLETENESS_STANDARD}_all.csv")
    # print(data_path)
    # exit()
    if os.path.exists(data_path):
        route_df = pd.read_csv(data_path)
    else:
        process_log(f"Will build dataset for {prefix}")
        route_df = get_related_route(prefix, collect_time)
    rov_count_distribute = route_df['has_rov'].value_counts()
    df_data_counts = pd.DataFrame(rov_count_distribute)
    
    print(df_data_counts)
    df_path = os.path.join(GNN_DATA_PATH, collect_time, f"{prefix_for_file}_{SERVE_COMPLETENESS_STANDARD}.csv")
    check_path_directory(df_path)
    df_data_counts.to_csv(df_path)
    
def multi_make_internet_rec_dataset(collect_time):

    process_log(f"Loading route data....")
    total_r_df = init_route(collect_time)
    # total_r_df = total_r_df[:400] # for test
    total_route_len = len(total_r_df)
    
    process_log(f"Try to recommend in {total_route_len} routes")
    # exit()
    node_embedding_path = os.path.join(GDS_DATA_PATH, "as_embedding.csv")
    node_embedding_df = pd.read_csv(node_embedding_path)
    as_series= node_embedding_df['as_num'].astype('str') 
    node_embedding_df = node_embedding_df.set_index("as_num")
    
    new_df = pd.DataFrame(as_series)
    new_df['count'] = 0
    new_df = new_df.set_index("as_num")
    print(new_df)
    
    model_path = os.path.join(GNN_MODEL_PATH, collect_time, "test_model.pth")
    process_log(f"Using model: {model_path}")
    model = torch.load(model_path)
    
    start_time = time.time()
    multi_core = 10
    d_len = total_route_len//multi_core
    mp = multiprocessing.Pool(multi_core)
    
    route_devided = []
    
    for i in range(0, multi_core-1):
        devide_data = total_r_df[i*d_len:(i+1)*d_len]
        route_devided.append(
            mp.apply_async(
                func=multi_rov_recommend,
                kwds={'data_df':devide_data, 'model':model, 'embedding_df':node_embedding_df, 'count_df': new_df.copy(), 'process_id':i}))
    devide_data = total_r_df[d_len*(multi_core-1):total_route_len]
    route_devided.append(
        mp.apply_async(
            func=multi_rov_recommend,
            kwds={'data_df':devide_data, 'model':model, 'embedding_df':node_embedding_df, 'count_df': new_df.copy(), 'process_id':i}))
    
    mp.close()
    mp.join()
    
    rec_dataset_list = []
    
    for result in tqdm(route_devided):
        part_res = result.get()
        rec_dataset_list.append(part_res)
    
    # total_rec_frame = pd.concat(rec_dataset_list, ignore_index=True)
    
    # result_rec_df = pd.DataFrame(total_rec_frame['rec'].value_counts())
    
    for df in rec_dataset_list:
        new_df = new_df.add(df)
    
    print(new_df)
    
    result_save_path = os.path.join(GNN_DATA_PATH, "rec", collect_time, "top_list.csv")
    check_path_directory(result_save_path)
    new_df.to_csv(result_save_path)
    end_time = time.time()
    return process_log(f"Recommendation on Internet ({collect_time}) Done, use time:{(end_time-start_time)/60} minutes")
    
def multi_rov_recommend(data_df, model, embedding_df, count_df, process_id):
    start_time = time.time()
    count = 0
    
    for index, row in data_df.iterrows():
        as_path = row['as_path']
        
        if "{" in as_path:
            continue
        
        as_list = as_path.split(',')
        as_path_len = int(row['path_len'])
        adj_matrix_nonzero = make_adj_matrix(as_path_len)
       
        x_feature = make_x_feature(as_list, embedding_df)
    
        # meanless_y
        y_label = [0 for i in range(len(as_list))]
        
        data = Data(x = x_feature, y=y_label, edge_index=adj_matrix_nonzero)
        output = model(data)
        pred = output.max(dim=1)[1].numpy()
        
        for i in range(as_path_len):
            as_num = as_list[i]
            rec_res = pred[i]
            if rec_res != 0:
                count_df.at[as_num, 'count'] += 1
        
        count += 1
        if count % 10000 == 0:
            print(f"Process {process_id}: Processd {count} data. Processing for {(end_time-start_time)/60} minutes\n")
        
    end_time = time.time()
    print(f"All done Process {process_id}: Processd {len(data_df)} data. Processing for {(end_time-start_time)/60} minutes\n")
    
    return count_df


def make_internet_rec_dataset(collect_time):

    # exit()
    node_embedding_path = os.path.join(GDS_DATA_PATH, "as_embedding.csv")
    node_embedding_df = pd.read_csv(node_embedding_path)
    as_series= node_embedding_df['as_num'].astype('str') 
    node_embedding_df = node_embedding_df.set_index("as_num")
    
    new_df = pd.DataFrame(as_series)
    new_df['count'] = 0
    new_df = new_df.set_index("as_num")
    print(new_df)
    process_log(f"Loading route data....")
    total_r_df = init_route(collect_time)
    # total_r_df = total_r_df[:400] # for test
    total_route_len = len(total_r_df)
    print(total_r_df)
    process_log(f"Try to recommend in {total_route_len} routes")
    
    model_path = os.path.join(GNN_MODEL_PATH, collect_time, "test_model.pth")
    process_log(f"Using model: {model_path}")
    model = torch.load(model_path)
    
    result_save_path = os.path.join(GNN_DATA_PATH, "rec", collect_time, "top_list.csv")
    protect_path = os.path.join(GNN_DATA_PATH, "rec", collect_time, "processing")
    if os.path.exists(protect_path):
        with open(protect_path, 'r') as f:
            context = f.read()
        if len(context) != 0:
            start_i = int(context)+1
            new_df = pd.read_csv(result_save_path)
            new_df['as_num'] = new_df['as_num'].astype('str') 
            new_df = new_df.set_index("as_num")
        else:
            start_i = 0
    else:
        start_i = 0
    process_log(f"From {start_i} to begin")
    start_time = time.time()
    
    for i in tqdm(range(start_i, len(total_r_df))):
        row = total_r_df.iloc[i]
        as_path = row['as_path']
        
        if "{" in as_path:
            continue
        
        as_list = as_path.split(',')
        as_path_len = row['path_len']
        adj_matrix_nonzero = make_adj_matrix(as_path_len)
       
        x_feature = make_x_feature(as_list, node_embedding_df)
    
        # meanless_y
        y_label = [0 for x in range(len(as_list))]
        
        data = Data(x = x_feature, y=y_label, edge_index=adj_matrix_nonzero)
        output = model(data)
        pred = output.max(dim=1)[1].numpy()
        
        for j in range(as_path_len):
            as_num = as_list[j]
            rec_res = pred[j]
            if rec_res != 0:
                new_df.at[as_num, 'count'] += 1
        
        if i % 50000 == 0:
            with open(protect_path, 'w') as f:
                f.write(str(i))
            new_df.to_csv(result_save_path)    
        
    result_df =  new_df.sort_values(by='count', ascending=False)
    
    check_path_directory(result_save_path)
    result_df.to_csv(result_save_path)
    end_time = time.time()
    return process_log(f"Recommendation on Internet ({collect_time}) Done, use time:{(end_time-start_time)/60} minutes")
