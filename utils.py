#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import inspect
import os, re
import IPy
import networkx as nx
import matplotlib.pyplot as plt
import calendar
import gzip, bz2

from global_settings import GRAPH_SAVE_PATH

from datetime import datetime

debug_level = 3

# util part
def find_pos(pattern, content):
    p = re.search(pattern, content)
    if (p == None):
        return None
    return p.start()


def read_file(file_name):
    with open(file_name, 'r') as f:
        file_text = f.read()
    return file_text

def process_log(text, level=2):
    caller_function_name = inspect.stack()[1][3]
    if (debug_level >= 3 and level == 3):
        print(f"\033[92m[#]Debug:\033[0m {text} \033[94m[{caller_function_name}]\033[0m")
    elif (debug_level >= 2 and level == 2):
        print(f"\033[94m[+]Logger:\033[0m {text} \033[94m[{caller_function_name}]\033[0m")
    elif (debug_level >= 1 and level == 1):
        print(f"\033[93m[!]Warning:\033[0m {text} \033[94m[{caller_function_name}]\033[0m")
    elif (debug_level >= 0 and level == 0):
        print(f"\033[91m[X]Error:\033[0m {text} \033[94m[{caller_function_name}]\033[0m")
    return 0


def save_to_file(content, filename, append=1):
    if (append):
        with open(filename, 'a+') as f:
            f.write(f"{content}")
    else:
        with open(filename, 'w') as f:
            f.write(f"{content}")


def change_work_path(work_path):
    if (os.path.isdir(work_path)):
        if (os.chdir(work_path)):
            process_log(f"Work path changes to {work_path}.", 3)
        else:
            process_log(f"Fail to change work path to {work_path}.", 0)
    else:
        process_log(f"{work_path} is not a valid path", 0)


def get_arg_from_list(arg_list):
    if len(arg_list) != 0:
        arg = arg_list.pop(0)
    else:
        arg = "?"
        process_log("Cannot get any more arg!", 0)
    return arg

def read_lines(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line

def split_path_into_dir_base(path):
    dirname, basename = os.path.split(path)
    return dirname, basename

def extract_gz_file(gz_file, extract_file):
    try:
        with gzip.open(gz_file, 'rb') as f_in, open(extract_file, 'wb') as f_out:
            f_out.write(f_in.read())
        return True
    except Exception as e:
        process_log(f"Error in gz xtracting: {e}", 0)
        return False
    
def extract_bz2_file(bz2_file, extract_file):
    try:
        with bz2.BZ2File(bz2_file, 'rb') as f_in, open(extract_file, 'wb') as f_out:
            f_out.write(f_in.read())
        return True
    except Exception as e:
        process_log(f"Error in bz2 xtracting: {e}", 0)
        return False

def read_compressed_file(file_path):
    if file_path.endswith('.bz2'):
        obj = bz2.BZ2File
    elif file_path.endswith('.gz'):
        obj = gzip.GzipFile
    else:
        return None
    f = obj(file_path, 'rb')
    return f.read()

def check_path_directory(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.isdir(directory):
        os.makedirs(directory)

def get_int_from_input(data):
    try:
        num = int(data)
        return True
    except:
        return False

# ----------------------------------------

# network part

def ip_mask_to_network(source_ip, source_mask, set_wild=-1):
    ip_part = [int(part) for part in source_ip.split('.')]
    mask_part = [int(part) for part in source_mask.split('.')]
    
    if (set_wild == 1):
        for i in range(4):
            mask_part[i] = 255 - mask_part[i]
        cidr = convert_subnet_mask_to_cidr(source_mask, 1)
    elif (set_wild == 0):
        cidr = convert_subnet_mask_to_cidr(source_mask, 0)
    else:
        if (mask_part[0] < mask_part[3]): 
            for i in range(4):
                mask_part[i] = 255 - mask_part[i]
            cidr = convert_subnet_mask_to_cidr(source_mask, 1)
        else:
            cidr = convert_subnet_mask_to_cidr(source_mask, 0)
    network_part = [str(ip & mask) for ip, mask in zip(ip_part, mask_part)]
    network = '.'.join(network_part)
    network_label = "%s/%d" % (network, cidr)
    return network_label

def convert_subnet_mask_to_cidr(subnet_mask, wild_mask): # if wild_mask, 32-cidr
    parts = subnet_mask.split('.')
    cidr = ''.join([bin(int(part))[2:].zfill(8) for part in parts]).count('1')
    if (wild_mask):
        cidr = 32 - cidr
    return cidr

def is_ip_in_network(ip_address, network):
    if network == None or ip_address == None:
        return False
    return ip_address in IPy.IP(network, make_net=1)

def is_prefix_in_another(prefix, another_prefix):
    return IPy.IP(prefix, make_net=1) in IPy.IP(another_prefix, make_net=1)

def unify_juniper_network(network):
    return str(IPy.IP(network, make_net=1))

def network_to_ip_mask(prefix, type=0):
    try:
        ip_addr = IPy.IP(prefix, make_net=1)
        ip = prefix.split('/')[0]
        mask = str(ip_addr.netmask())
    except Exception as e:
        process_log(f"Invalid ip! {e}", 0)
        return (None, None)
    
    
    if type == 0:
        return (ip, convert_subnet_mask_to_cidr(mask, 0))
    else:
        return (ip, mask)

def int_to_dot_decimal(number):
    parts = []
    while number > 0:
        parts.append(str(number % 256))
        number //= 256
    remain = 4 - len(parts)
    for i in range(remain):
        parts.append('0')
    return '.'.join(reversed(parts))

def check_prefix_format(prefix):
    try:
        ip_addr = IPy.IP(prefix, make_net=1)
        return True
    except Exception as e:
        process_log(f"Invalid prefix format: {prefix}! {e}", 0)
        return False
    
def compare_as_path(path1, path2):
    if path1 == None:
        return path2
    else:
        if path2 == None:
            return path1
        else:
            if len(path1) < len(path2): # choose shorter
                return path1
            else:
                if len(path1) == len(path2): # choose smaller as
                    int1_list = list(map(int, path1))
                    int2_list = list(map(int, path2))
                    if int1_list <= int2_list:
                        return path1
                    else:
                        return path2
                else:
                    return path2 
                
# ----------------------------------------

# networkx part

def build_graph_by_edge(edge_list):
    graph = nx.Graph()
    graph.add_edges_from(edge_list)
    return graph

def draw_graph(G):
    pos = nx.spring_layout(G)
    # Draw the graph using the computed positions
    nx.draw(G, pos, with_labels=True, node_size=500, alpha=0.6, connectionstyle='arc3, rad = 0.25', arrows=True)

    # Set plot title and show the plot
    plt.title("Graph Topology")
    plt.show()

def save_graph(G, file_name):
    if not os.path.exists(GRAPH_SAVE_PATH):
        os.mkdir(GRAPH_SAVE_PATH)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500)

    # Set plot title and show the plot
    plt.title("Graph Topology")
    plt.savefig(os.path.join(GRAPH_SAVE_PATH, file_name))
    plt.close() 


# ----------------------------------------

import dill

class IncrementalSaver:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        os.makedirs(self.folder_path, exist_ok = True)
        self.latest = self.find_latest() 
    
    def find_latest(self):
        # files = os.listdir(self.folder_path)
        counter = 0
        while 1:
            file_path = os.path.join(self.folder_path,f"{counter}.pkl")
            if not os.path.exists(file_path):
                break
            counter += 1
        return counter-1
    
    def write(self, obj):
        
        self.latest += 1
        file_path = os.path.join(self.folder_path,f"{self.latest}.pkl")
        
        print(f"write to {file_path}")
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    
    def read_dict(self):
        result = {}
        for i in range(self.latest+1):
            file_path = os.path.join(self.folder_path,f"{i}.pkl")
            with open(file_path, 'rb') as file:
                result.update(dill.load(file))
        return result
    
    def read_dataframe(self): # for overwrite

        file_path = os.path.join(self.folder_path,f"{self.latest}.pkl")
        with open(file_path, 'rb') as file:
            df = dill.load(file)
        return df
    
    def append(self, obj):

        file_path = os.path.join(self.folder_path,f"{self.latest}.pkl")
        print(f"append to {file_path}")
        with open(file_path, 'ab') as file:
            dill.dump(obj, file)

    def overwrite(self, obj):

        if self.latest < 0:
            self.latest = 0
        file_path = os.path.join(self.folder_path,f"{self.latest}.pkl")
        print(f"overwrite to {file_path}")
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

# ----------------------------------------

# time part

def is_start_smaller_than_end(start_time, end_time):
    s_t = datetime.strptime(start_time, '%Y%m%d')
    e_t = datetime.strptime(end_time, '%Y%m%d')
    return s_t <= e_t

def is_valid_time_period(date_str):
    date_list = date_str.split('-')
    if len(date_list) != 2:
        return False
    else:
        start_time = date_list[0]
        end_time = date_list[1]
        if (not is_valid_yyyymmdd(start_time)) or (not is_valid_yyyymmdd(end_time)):
            return False
        else:
            return is_start_smaller_than_end(start_time, end_time)

def is_valid_yyyymmdd(date_str):
    try:
        datetime.strptime(date_str, '%Y%m%d')
        return True
    except ValueError:
        return False
    
def get_YYYYMMDD_from_filename(file_path):
    filename = os.path.basename(file_path)
    pattern = r".*(\d{8}).*"
    date_time = re.search(pattern, filename)
    return date_time.group(1)

def is_valid_time_set(data_time):
    for t in data_time:
        if '-' in t:
            if not is_valid_time_period(t):
                process_log(f"Invalid time format '{t}'! Please input in correct format.", 0)
                return False
        else:
            if not is_valid_yyyymmdd(t):
                process_log(f"Invalid time format '{t}'! Please input in correct format.", 0)
                return False
    return True

def get_every_day_from_time(date):
    date_list = date.split('-')
    begin_date = date_list[0]
    end_date = date_list[1]
    s_t = datetime.strptime(begin_date, '%Y%m%d')
    e_t = datetime.strptime(end_date, '%Y%m%d')
    result = []
    while s_t <= e_t:  
        date_str = s_t.strftime("%Y%m%d")
        result.append(date_str)  
        s_t = add_days(s_t,1)
    
    return result  

def add_days(dt,days): 
    day = dt.day + days
    month = dt.month  
    year = dt.year
    if day > calendar.monthrange(year, month)[1]:
        day -= calendar.monthrange(year, month)[1]
        month += 1 
        if month > 12:
            year += 1
            month = 1
    return dt.replace(year=year, month=month, day=day)  

def get_quarter_date_first_month(date):
    month = date.month
    if 1 <= month <= 3:
        return 1
    elif 4 <= month <= 6:
        return 4
    elif 7 <= month <= 9:
        return 7
    else:
        return 10
    
  
def add_months(dt,months): 
    month = dt.month - 1 + months  
    year = dt.year + month // 12  
    month = month % 12 + 1  
    day = min(dt.day, calendar.monthrange(year, month)[1])  
    return dt.replace(year=year, month=month, day=day)  