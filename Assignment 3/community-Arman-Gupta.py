import pandas as pd
import numpy as np
from matplotlib import cm
from collections import defaultdict
from itertools import combinations 
import numpy as np
float_formatter = lambda x: "%.15f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
# from sklearn.datasets.samples_generator import make_circles
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
import networkx as nx
import seaborn as sns
from tqdm import tqdm
import datetime, time
import warnings
warnings.filterwarnings("ignore")
sns.set()
tolerance = 5e-14
np.random.seed(0)

## Data Loading 
def import_facebook_data(path):
    f = open(path,'r')
    edges = np.array(f.read().split('\n'))
    edges = [list(map(int , edge.split())) for edge in edges if len(edge.split()) == 2 ] 
    rev_edges = [[edge[1],edge[0]] for edge in edges]
    edges.extend(rev_edges)
    edges = np.array(edges)
    return edges

def import_bitcoin_data(path):
    bitcoin = pd.read_csv(path,names=['SOURCE', 'TARGET', 'RATING', 'TIME'])
    bitcoin = bitcoin.sort_values(['SOURCE','TARGET'],ascending= [True,True]).reset_index(drop=True)
    nodes = list(set(bitcoin['SOURCE']).union(bitcoin['TARGET']))
    nmapper = dict(zip(nodes , np.arange(len(nodes))))

    bitcoin['SOURCE'] = bitcoin['SOURCE'].apply(lambda x :  nmapper[x])
    bitcoin['TARGET'] = bitcoin['TARGET'].apply(lambda x :  nmapper[x])
    edges  = bitcoin[['SOURCE','TARGET']].to_numpy()
    rev_edges = np.array([[edge[1],edge[0]] for edge in edges])

    edges = np.append(edges , rev_edges ,axis = 0)
    edges = np.array(edges)
    return edges

## ************************* SPECTRAL CLUSTERING ******************************
# Functions for drawing Plots 
def drawAdjacency1(A , name):
    nnodes = len(A)
    u_edges = []
    for i in range(nnodes):
        for j in range(nnodes):
            if A[i][j] == 1:
                u_edges.append([i,j])
    u_edges = np.array(u_edges)
    fig = plt.figure(figsize=(10, 10)) # in inches
    plt.scatter(u_edges[:, 0],u_edges[:, 1], s=1)
    plt.savefig('./Plots/'+name+'.jpeg')
def drawAdjacency(nodes, edges):
    n_mapper = dict(zip(nodes, np.arange(len(nodes))))
    u_edges = np.array([[n_mapper[edge[0]] , n_mapper[edge[1]]] for edge in edges])

    fig = plt.figure(figsize=(10, 10)) # in inches
    plt.scatter(u_edges[:, 0],u_edges[:, 1], s=1)
    plt.show()

def drawGraph(nodes , edges , values):
    fig = plt.figure(figsize=(10, 10)) 
    n_mapper = dict(zip(nodes, np.arange(len(nodes))))
    u_edges = np.array([[n_mapper[edge[0]] , n_mapper[edge[1]]] for edge in edges])
    G = nx.Graph()
    G.add_edges_from(u_edges)
    nx.draw(G , node_color = values)
    plt.show()

def drawGraphWithClusters(nodes , clusters, edges ,values):
    n_mapper = dict(zip(nodes, np.arange(len(nodes))))
    u_edges = []
    plt.figure(figsize=(10,10))
    color = iter(cm.plasma_r(np.linspace(0, 1, len(clusters))))
    G = nx.Graph()
    G.add_edges_from(edges )
    nx.draw(G, node_color = values)


## One Iteration of Spectral Clustering
def spectralClustering(nodes , edges , drawplots = False):
    '''
        nodes : 1D array contain ordered list of nodes
        edges : edge list of size |E| * 2
        drawplots : if True plot sorted feidler vector, Adjacency list and Graph partition
    '''
    nnodes = len(nodes)
    n_mapper = dict(zip(nodes, np.arange(nnodes)))

    A = np.zeros((nnodes,nnodes))
    for edge in edges:
        if edge[0] in n_mapper and edge[1] in n_mapper:
            A[n_mapper[edge[0]],n_mapper[edge[1]]] = 1
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    eval, evec = np.linalg.eigh(L)
    index = (eval <= tolerance).sum()
    fv = evec[:,index]
    i_sort_fv = fv.argsort()
    sorted_fv = fv[i_sort_fv]

    u_nodes = nodes[i_sort_fv]
    if drawplots:
        fig = plt.figure(figsize=(10, 10)) # in inches
        plt.scatter(np.arange(sorted_fv.shape[0]),sorted_fv)
        plt.savefig('./Plots/q1_SortedFiedler.png')
        values = np.append(np.zeros(np.sum(sorted_fv < 0)),np.ones(np.sum(sorted_fv >= 0)))
        drawAdjacency(u_nodes , edges)
        drawGraph(u_nodes, edges ,values= values )
    return fv , sorted_fv , i_sort_fv , u_nodes

## Algorithm for finding cluster using Spectral clustering automatically based on
## modularity change.
def autoFindClusters(nodes, edges):
    '''
        nodes : 1D array contain ordered list of nodes
        edges : edge list of size |E| * 2
    '''
    clusters = []
    stack =[]
    stack.append(nodes)
    nnodes  = len(nodes)
    A = np.zeros((nnodes,nnodes))
    for edge in edges:
        A[edge[0],edge[1]] = 1

    while(len(stack) > 0):
        cur_nodes = stack.pop()
        fv , sorted_fv , argsort_fv , u_nodes = spectralClustering(cur_nodes,edges)
        mean = 0
        modularity_before = computeModularity(cur_nodes , A)
        nodes1 = u_nodes[sorted_fv <= mean]
        nodes2 = u_nodes[sorted_fv > mean]
        modularity_after = computeModularity(nodes1 , A) + computeModularity(nodes2, A)
        print(len(nodes1) , len(nodes2), modularity_before, modularity_after)

        if(modularity_after <= modularity_before):
            clusters.append(u_nodes)
        else:
            stack.append(nodes1)
            stack.append(nodes2)
    return clusters


## Implementation of Function based on Template.
## Use spectralClustering() method internally and create the output according to given template
def spectralDecomp_OneIter(edges):
    '''
     edges : edge list of size |E| * 2
     return :
      1. feidler vector of size |V| * 1 (unsorted)
      2. adjacency matrix of actual graph of size |V| * |V|  (unsorted)
      3. graph partitions with community Id as min id of node in community
    '''
    nnodes  = np.max(edges) + 1
    nodes = np.arange(nnodes)
    fv , sorted_fv , i_sort_fv , u_nodes = spectralClustering(nodes , edges , drawplots= False)
    n_mapper = dict(zip(u_nodes, np.arange(len(u_nodes))))
    u_edges = np.array([[n_mapper[edge[0]] , n_mapper[edge[1]]] for edge in edges])

    A = np.zeros((nnodes ,nnodes))
    for edge in u_edges:
        A[edge[0]][edge[1]] = 1
    
    nodes1 = u_nodes[sorted_fv <= 0]
    nodes2 = u_nodes[sorted_fv > 0]
    partitioned_nodes = [nodes1 , nodes2]
    graph_partition = np.zeros((nnodes , 2))

    k = 0
    for i,cluster in enumerate(partitioned_nodes):
        com_id = np.min(cluster)
        for node in cluster:
            graph_partition[k][0] = node
            graph_partition[k][1] = com_id
            k+=1
    return fv , A , np.array(graph_partition)

## Implementation of Function based on Template.
## Use autoFindClusters() function internally and create the output according to given template
def spectralDecomposition(edges):
    '''
      edges : edge list of size |E| * 2
      return : graph partitions of size |V| * 2 with community Id as min id of node in community
    '''
    nnodes  = np.max(edges) + 1
    nodes = np.arange(nnodes)
    clusters = autoFindClusters(nodes ,edges)
    graph_partition = np.zeros((nnodes , 2))
    print(f'Number of Clusters using spectral decompositions are {len(clusters)}')

    k = 0
    for i, cluster in enumerate(clusters):
        com_id = np.min(cluster)
        for node in cluster:
            graph_partition[k][0] = node
            graph_partition[k][1] = com_id
            k+=1
    return graph_partition

## Implementation of Function based on Template.
def createSortedAdjMat(graph_partition , edges):
    '''
      graph_partition : graph partitions of size |V| * 2 
      edges : edge list of size |E| * 2
      return : Sorted Ajacency Matrix
    '''
    clusters = defaultdict(list)
    for pair in graph_partition:
        clusters[pair[1]].append(pair[0])

    nodes = np.array([])
    for k,v in clusters.items():
        nodes = np.append(nodes, clusters[k])

    n_mapper = dict(zip(nodes, np.arange(len(nodes))))
    u_edges = np.array([[n_mapper[edge[0]] , n_mapper[edge[1]]] for edge in edges])
    nnodes = len(nodes)

    A = np.zeros((nnodes ,nnodes))
    for edge in u_edges:
        A[edge[0]][edge[1]] = 1
    return A


# ***************************** Louvain  Algorithm **************************************

## Compute sigma in (as per convention followed in lecture)
def comLinkWeightsSum(community, A):
    sum = 0
    inedges = list(combinations(community,2)) + [(node, node) for node in community]
    for e in inedges:
        sum += 2*A[e[0]][e[1]]
    return sum

## Compute the sum of degree of all nodes in the community
def comDegreeSum(community, A):
    return np.sum(A[community])

## compute degree of vertex v considering only the community 
def get_Ki_in(community, v, A, isAdd = True):
    A_v = A[v]
    if isAdd:
        A_v_com = A_v[community]
    else:
        A_v_com = A_v[np.setdiff1d(community , [v])]
    return 2 * np.sum(A_v_com)

## compute modularity of the community
def computeModularity(community, A):
    m = np.sum(A)
    sig_in = comLinkWeightsSum(community , A)
    sig_tot = comDegreeSum(community, A)
    return sig_in / (2*m) - (sig_tot / (2*m)) **2 

## create dictionary from community to node lists
def createCom2Nodes(node2com):
    com2nodes = defaultdict(list)
    for node ,community_id in  node2com.items():
        com2nodes[community_id].append(node)
    return com2nodes

## compute sum of the modularity of all clusters.
def computeModularityClusters(node2com , A):
    '''Return vector of size |comunities| indicating modularity of clusters'''
    com2nodes = createCom2Nodes(node2com)

    com_modularity_sum = 0
    for com_id, community in com2nodes.items():
        com_modularity_sum += computeModularity(community , A)
    return com_modularity_sum

## Compute delta modularity
def deltaModularity(community , i , A , isAdd = True):
    if isAdd:
        if i in community : 
            return 0
        q_before  = computeModularity(community , A) + computeModularity([i],A)
        q_after = computeModularity(np.append(community,[i]),A)
        
    else:
        if i not in community:
            return 0
        q_before = computeModularity(community , A)
        q_after = computeModularity(np.setdiff1d(community , [i]) , A) + computeModularity([i],A)

    return q_after - q_before

## compute neighbour nodes of certain vertex
def neighbourNodes(v, A):
    return np.where(np.array(A[v]) != 0 )[0]  


## 1 iteration of Phase 1 
def phase1_iter(node2com, A):
    m = np.sum(A)
    node2com_copy = node2com.copy()
    com2nodes = createCom2Nodes(node2com)
    is_changed = False
    node_degrees = np.sum(A , axis = 1)
    for v in tqdm( node2com.keys() ):
        cur_com_id = node2com[v]
        max_delta_q = - np.inf
        max_com_id = cur_com_id
        cur_community = com2nodes[cur_com_id]

        neighbours = neighbourNodes(v,A)
        neigh_com_ids = np.unique(list(map(node2com.get , neighbours)))
        delta_q_remove_node_v = -get_Ki_in(cur_community , v , A , False) /(2*m) + comDegreeSum(cur_community,A) * node_degrees[v] / m \
            - 2 * (node_degrees[v]/(2*m))**2
        for neigh_com_id in neigh_com_ids:
            if neigh_com_id == cur_com_id :
                continue
            community = com2nodes[neigh_com_id] 
            delta_q = get_Ki_in(community , v, A , True) / (2*m) - comDegreeSum(community,A) * node_degrees[v] / m +delta_q_remove_node_v
            if delta_q > max_delta_q:
                max_delta_q = delta_q
                max_com_id = neigh_com_id
        if cur_com_id != max_com_id:
            is_changed = True
        node2com_copy[v] = max_com_id

    return node2com_copy , is_changed 

## Internally calling phase1_iter until either no change in modularity 
# or the patience  is over
def phase1(node2com , A , patience_ = 6):
    is_changed = True
    node2com_copy = node2com.copy()
    print('Patience is ' , patience_)
    max_modularity = computeModularityClusters(node2com_copy, A)
    print('Modularity : ', max_modularity)
    best_node2com = node2com.copy()
    patience = patience_
    ## If the total modularity of the Graph with the assigned cluster does
    #  not increase for fixed patience number of the steps , the algorithm
    #  stops and give the communities at best modularity.
    while is_changed :
        node2com_copy ,is_changed =  phase1_iter(node2com_copy , A)
        cur_modularity =  computeModularityClusters(node2com_copy, A)
        print('Modularity : ', cur_modularity) 
        if cur_modularity > max_modularity:
            best_node2com = node2com_copy.copy()
            max_modularity = cur_modularity
            patience = patience_
        else:
            patience -=1
        if patience == 0:
            break
    return best_node2com , max_modularity
    
## Implemented this function just to draw the community graph
def phase2(node2com , A):
    nnodes = len(A)
    com_ids = sorted(np.unique(list(node2com.values())))
    super_nnodes = len(com_ids)
    com_mapper = dict(zip(com_ids , np.arange(super_nnodes))) # Since  com_ids can be non contiguous
    rev_com_mapper = dict(zip(np.arange(super_nnodes) , com_ids))
    superA = np.zeros((super_nnodes , super_nnodes))
    superEdges = []
    for i in range(nnodes):
        for j in range(nnodes):
            if A[i ,j] == 1:
                com_id_i = com_mapper[node2com[i]]
                com_id_j = com_mapper[node2com[j]]
                superA[com_id_i ][ com_id_j] +=1
    for i in range(super_nnodes):
        for j in range(super_nnodes):
            if superA[i][j]!=0:
                superEdges.append([i , j])
    return superA  , np.array(superEdges), rev_com_mapper  


## Implementation of function according to template
def louvain_one_iter(edges):
    nnodes  = np.max(edges) + 1
    A = np.zeros((nnodes,nnodes))
    for edge in edges:
        A[edge[0],edge[1]] = 1
    nodes = np.arange(nnodes)
    node2com = dict(zip(nodes, np.arange(nnodes)))
    best_node2com , max_modularity =  phase1(node2com , A , patience_= 20)
    
    print(f'Best Modularity : {max_modularity} and number of communities : {np.unique(list(best_node2com.values())).shape}' )

    com2nodes = createCom2Nodes(best_node2com)
    graph_partition = np.zeros((nnodes, 2))
    k = 0
    for key, cluster in com2nodes.items():
        com_id = np.min(cluster)
        for node in cluster:
            graph_partition[k][0] = node
            graph_partition[k][1] = com_id
            k+=1
    
    return graph_partition

if __name__ == "__main__":

    ############ Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is a edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")

    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)

    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # identified as part of question 2. The naming convention for the community id is as before.
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)

    # This is for question no. 3
    # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # adjacency matrix is to be sorted in an increasing order of communitites.
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before.
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)


    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import soc-sign-bitcoinotc.csv
    nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")

    # Question 1
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)

    # Question 2
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)

    # Question 3
    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)

    # Question 4
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)
