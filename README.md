# ROVReco
**(The source code of ROVReco is currently under maintenance. It will be uploaded once the maintenance is completed.)**

Here is the available data of ROVReco. ROVReco is a framework which aims to recommend the Autonomous Systems (ASes) to deploy Route Origin Validation (ROV) process.

These codes will help you to bulid the knowledge graph (KG) of Internet Routing System, which supports the recommendation process of ROVReco. The relevant code of ROVReco will be gradually updated for public use.


## Current Measurement Data

Currently, we have conducted ROVReco measurements for the Internet as of July 1, 2024. We conducted a routing analysis on the following prefixes and, based on the recommended experiences derived from them, expanded the recommendation for the ROV deployment strategy across the entire network.

18.65.8.0/22 

59.71.48.0/22

168.167.71.0/24

195.54.164.0/23

81.90.208.0/20

45.160.192.0/22

144.208.192.0/22

102.68.115.0/24

The specific information of these prefixes (such as the owning AS number, the region, etc.) can be queried on websites like https://bgp.he.net/ and https://www.radb.net/. 

When we randomly selected these eight prefixes, we tried to cover regions all over the world to encompass various network environments. This proves the effectiveness of ROVReco in both the small-scale analysis (at the prefix level) and the large-scale analysis (across the entire internet).

## Used Publicly Available Dataset

You can collect the public datasets from anytime you like. The needed datasets are listed as follow:

### BGP Routing Data. 
Route Views and RIPE RIS have the largest BGP monitoring systems in the industry and provide the most comprehensive and authoritative BGP monitoring data. In this paper, we use BGP RIB data from all collectors to construct the KG, supporting our research on ROV deployment at the routing level.
RouteViews：http://archive.routeviews.org/
RIPE RIS：https://data.ris.ripe.net/

### CAIDA Dataset. 
We utilized three publicly available datasets sourced from the Center for Applied Internet Data Analysis (CAIDA): AS Relationships, AS to Organization (AS2Org) and Prefix to AS (Prefix2AS). We use them to jointly characterize the basic features of AS on the Internet.
AS Relationships：https://publicdata.caida.org/datasets/as-relationships/serial-1/
AS to Organization (AS2Org)：https://publicdata.caida.org/datasets/as-organizations/
Prefix to AS (Prefix2AS)：https://publicdata.caida.org/datasets/routing/routeviews-prefix2as/ (IPv4)
                          https://publicdata.caida.org/datasets/routing/routeviews6-prefix2as/ (IPv6)

### RPKI Repository Archives. 
RPKI resources are initially distributed by the Internet Assigned Numbers Authority (IANA) to Regional Internet Registries (RIRs), which then redistribute them to Local Internet Registries (LIRs), and ultimately to their customers. The RIPE NCC website provides daily summaries of ROAs from five RIR trust anchors. 
RPKI state：https://ftp.ripe.net/rpki/


## Running

We run ROVReco on a Linux server with two Intel(R) Xeon(R) Silver 4210R x86_64 2.40GHz CPUs and 125GB main memory. ROVReco took 3-4 hours to bulid the KG.
The size of KG database is about 5-6 GB.

Finally, ROVReco provide a top list of ASes for ROV deployment recommendations on the Internet. Here we have used the datasets from 20240701 to give the recommendation.

### Knowledge Graph of Internet Routing System

We model the Internet Routing System using Neo4j.
The community version of Neo4j is fully capable of supporting our modeling. In ROVReco, we used the Neo4j Community Edition version 5.21.2 for the construction. You can build a knowledge graph by using the following command.
Please replace {kg_date} with the `YYYYMMDD` date and run the command in the corresponding storage root directory:

<code>neo4j-admin database import full --nodes {kg_date}/as_entity.csv --nodes {kg_date}/org_entity.csv --nodes {kg_date}/nic_entity.csv --nodes {kg_date}/roa_entity.csv --nodes {kg_date}/prefix_entity.csv  --relationships {kg_date}/as_relation.csv --relationships {kg_date}/org_relation.csv --relationships {kg_date}/nic2roa_relation.csv --relationships {kg_date}/prefix2as_relation.csv --relationships {kg_date}/roa2as_relation.csv --relationships {kg_date}/as_betweenness_relation.csv --skip-bad-relationships --overwrite-destination --verbose neo4j</code>

The download links for Neo4j-related software can be found at https://neo4j.com/deployment-center/#enterprise.

### Try the Demo Model

We have uploaded several models created using the prefix ROV recommendation experience training. The training data for the prefix was obtained from the ROVReco simulation, and the results can be found in the `preprocess_dataset/analyse_prefix` folder. 

We have simultaneously uploaded the AS embeddings obtained from the knowledge graph, which are put in the `gds_data` folder. Please decompress it before running ROVReco. There is a file named `global_settings.py` in the code file, where you can find information regarding the code path and data path issues.

You can try the recommendation process by running the `gnn_recommend_demo.ipynb` Jupyter script.


## Paper
The paper titled "ROVReco: An ROV Deployment Recommendation Approach with GNN Based on Routing Betweenness" regarding the work of ROVReco has been available at SSRN: https://ssrn.com/abstract=5170823 or http://dx.doi.org/10.2139/ssrn.5170823