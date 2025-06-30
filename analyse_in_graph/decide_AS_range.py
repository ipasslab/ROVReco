#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from neomodel import db, config
from neomodel.integration.pandas import to_dataframe
from analyse_in_graph.model_for_neomodel import *

from global_settings import *

from utils import *

import time
import multiprocessing
import pandas as pd
import tqdm

def find_prefix_of_ROA(ROA_prefix):
    find_prefix_node_key = IPy.IP(ROA_prefix, make_net=1).strNormal(0).split('0')[0].rstrip('.')[:-1]
    print(find_prefix_node_key)
    find_prefix_cypher = f"MATCH (p:Prefix) WHERE p.prefix CONTAINS '{find_prefix_node_key}' RETURN p.prefix as prefix"
    # result, meta = db.cypher_query(find_prefix_cypher)
    # print(result)
    def judge_prefix_in_roa(prefix):
        return is_prefix_in_another(prefix, ROA_prefix)
    result_df = to_dataframe(db.cypher_query(find_prefix_cypher))
    ROA_affect_prefix = result_df[result_df['prefix'].apply(judge_prefix_in_roa)]
    # print(ROA_affect_prefix)
    return ROA_affect_prefix

def measure_prefix_route_range(prefix):
    find_as_cypher = f"MATCH (a:AS)-[r:route]->(p:Prefix),(a:AS)-[b:belong]->(o:Organization), e=(:AS)-[:belong]->(o:Organization) \
        WHERE p.prefix = '{prefix}' \
        RETURN a.as as AS, r.value as betweenness, o.org_name as org_name"
    result_df = to_dataframe(db.cypher_query(find_as_cypher))
    # print(result_df)
    return result_df

def count_betweenness_total_of_prefix():
    pass

def get_org_AS(org_name):
    
    find_cypher = f"MATCH (a:AS)-[b:belong]->(o:Organization) WHERE o.org_name = '{org_name}' RETURN a.as as AS"
    result_df = to_dataframe(db.cypher_query(find_cypher))
    # print(result_df)
    return result_df
    
def get_basic_topology_of_AS_by_prefix(prefix):
    
    find_cypher = f"MATCH (a:AS)-[ar:route]->(p:Prefix), (b:AS)-[br:route]->(p:Prefix), i=(a:AS)-[r]->(b:AS)\
        WHERE p.prefix='{prefix}' \
        RETURN DISTINCT a.as as as1, type(r) as type, b.as as as2"
    result_df = to_dataframe(db.cypher_query(find_cypher))
    # print(result_df)
    return result_df

def get_listofAS_hasrouteto_prefix(prefix):
    find_cypher = f"MATCH (a:AS)-[ar:route]->(p:Prefix) \
        WHERE p.prefix='{prefix}' \
        RETURN DISTINCT a.as as as_num "
    result_list, meta = db.cypher_query(find_cypher)
    return [row[0] for row in result_list]

def get_AS_of_prefix(prefix):
    find_cypher = f"MATCH (a:AS)-[ar:route]->(p:Prefix), (b:AS)<-[br:prefix]-(p:Prefix)\
        WHERE p.prefix='{prefix}' AND ar.nexthop='self' \
        RETURN DISTINCT a.as as RIB_info, b.as as CAIDA_info"
    result_df = to_dataframe(db.cypher_query(find_cypher))
    return result_df

def check_AS_prefix(as_prefix_df):
    result_len = len(as_prefix_df)
    if result_len == 1:
        rib_as = as_prefix_df['RIB_info'].iloc[0]
        caida_as = as_prefix_df['CAIDA_info'].iloc[0]
        if rib_as == caida_as:
            return str(rib_as)
        else:
            return None
    else:
        return None

# ----------------------------------------

def get_AS_range_by_route(as_list):
    cypher = f"MATCH i=(a:AS)-[:route]->(p:Prefix)-[:prefix]->(b:AS) \
        WHERE a.as in {as_list}\
        RETURN DISTINCT b.as as as_num, p.prefix as prefix"
    # print(cypher)
    result_df = to_dataframe(db.cypher_query(cypher))
    
    return result_df

def get_customer_AS_range_by_route(as_list):
    cypher = f"MATCH i=(a:AS)-[:route]->(p:Prefix)-[:prefix]->(b:AS) \
        WHERE a.as in {as_list} AND NOT ( ()<-[:P2C]-(b) ) \
        RETURN DISTINCT b.as as as_num, p.prefix as prefix"
    
    result_df = to_dataframe(db.cypher_query(cypher))
    result_df['as_num'] = result_df['as_num'].astype(int)
    return result_df

def mr_get_AS_function(as_list):
    result_list = []
    for as_num in as_list:
        result_list.append(get_AS_range_by_route(as_num))
    return result_list
    
def multi_process_get_AS_range(as_list): # cannot use in neomodel
    start_time = time.time()
    as_count = len(as_list)
    result_list = []
    print(str(as_list))
    # result_list = mr_get_AS_function(as_list)

    # if as_count < MULTI_CORE_NUM:
    #     multi_core = as_count
    # else:
    #     multi_core = MULTI_CORE_NUM
    # d_len = as_count//multi_core
    # mp = multiprocessing.Pool(multi_core)
    # mission_devided = []
    # for i in range(0, multi_core-1):
    #     devide_data = as_list[i*d_len:(i+1)*d_len]
    #     mission_devided.append(
    #         mp.apply_async(
    #             func=mr_get_AS_function,
    #             kwds={'as_list':devide_data}))
    # devide_data = as_list[d_len*(multi_core-1):as_count]
    # mission_devided.append(
    #     mp.apply_async(
    #         func=mr_get_AS_function,
    #         kwds={'as_list':devide_data}))
    
    # mp.close()
    # mp.join()
    
    # for result in tqdm(mission_devided):
    #     result_list.extend(result.get())
    # as_df = pd.concat(result_list, ignore_index=True)
    # as_df = as_df.drop_duplicates()
    # print(as_df)
    # end_time = time.time()
    
    process_log("Decide as range over. Processing for {} minutes".format((end_time-start_time)/60))
    return as_df
# ----------------------------------------

# for futher analysis

def get_tier_AS(tier_num):
    if tier_num == 1:
        cypher = "MATCH (n:AS) WHERE NOT (n)<-[:P2C]-() return n.as as as_num"
        
    elif tier_num == 2:
        cypher = "MATCH (n:AS) WHERE ()<-[:P2C]-(n) and (n)<-[:P2C]-() return n.as as as_num"
    
    elif tier_num == 3:
        cypher = "MATCH (n:AS) WHERE not ()<-[:P2C]-(n) return n.as as as_num"
        
    result_df = to_dataframe(db.cypher_query(cypher))
    
    return result_df