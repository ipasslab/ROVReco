#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os

HISTORY_PROCESS_PATH = "processed"  # in configure's path
GRAPH_SAVE_PATH = "./graphs"
ROUTING_TABLE_SAVE_PATH = "./routing_table"

NEO4J_HOME = "......./neo4j-community-5.21.2" # absolute path
NEO4J_VERSION = 5
NEO4J_URL = "bolt://localhost:7687"
NEO4J_AUTH = ("username", "xxxxxxxxxxx")
DEFAULT_DATABASE = "neo4j"
WAIT_RESTART_TIME = 30

DEFAULT_BACKUP_NAME = "neo4j.dump.db"
DEFAULT_BACKUP_PATH = os.path.join(NEO4J_HOME, "backup", DEFAULT_BACKUP_NAME)



# new global

RPKI_CODE_BASE_PATH = '........'

# under RPKI_CODE_BASE_PATH
DATASET_PATH = './dataset/'

CAIDA_SAVE_PATH = './dataset/CAIDA/'

ROUTE_VIEWS_RIB_SAVE_PATH = './dataset/RIB/route_views/'

RIPE_RIS_RIB_SAVE_PATH = './dataset/RIB/ripe_ris/'

RPKI_SAVE_PATH = './dataset/RPKI/'

PREPROCESS_DATASET_PATH = './preprocess_dataset/'

ORIGIN_ROUTE_SAVE_PATH = './origin_route_path/'

PROGRESS_SAVE_PATH = './progress/'

ROVISTA_SAVE_PATH = './dataset/ROV/'

# for multi process
MULTI_CORE_NUM = 20

# for downloader
DOWNLOAD_OVER_SIGNFILE = "SUCC"
EXTRACT_OVER_SIGNFILE = "EXTRACTED"

# for builder
PROCESS_OVER_SIGNFILE = "PROCESSED"

# for analyse
ANALYSE_PROCESS_PATH = os.path.join(PREPROCESS_DATASET_PATH, 'analyse_prefix')

# for ROV sim
SERVE_COMPLETENESS_STANDARD = 0.7
ANALYSE_PROCESS_RATE_FILE = "PROCESS"


# for GDS analyse
GDS_DATA_PATH = "./gds_data/"

GNN_DATA_PATH = "./gnn_data/"

GNN_MODEL_PATH = "./gnn_model/"

GNN_TRAIN_EPOCH = 50


INTERNET_KG_PATH = "./preprocess_dataset/database/"