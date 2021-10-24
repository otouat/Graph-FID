import networkx as nx
import numpy as np
import random
from numpy.random import randint
import torch
from sklearn.model_selection import StratifiedKFold
import os
import pickle 
class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array(
                        [float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag]
                                                  for tag in g.node_tags]] = 1

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)


def load_synth_data(degree_as_tag,random):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''
    os.makedirs('saved_graphs', exist_ok=True)
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}
    mapped = len(label_dict)
    graph_list=[]
    label_dict[1] = mapped
    
    for k in range(100,200):
        for j in range(10):
            i = randint(1,10)*0.1
            g=nx.fast_gnp_random_graph(k,i)
            graph_list.append(g)
            g_list.append(S2VGraph(g, 1, [0] * g.number_of_nodes()))
    save_graph_list(
                graph_list,
                os.path.join('saved_graphs', '{}_test.p'.format('erdos')))
    graph_list=[]
    mapped = len(label_dict)
    label_dict[2] = mapped
        #c_sizes = [15] * 4
    for k in range(100,200):
        for j in range(10):
            g=nx.watts_strogatz_graph(k,4,0.1)
            graph_list.append(g)
            g_list.append(S2VGraph(g, 2, [0] * g.number_of_nodes()))
    save_graph_list(
                graph_list,
                os.path.join('saved_graphs', '{}_test.p'.format('watts')))
    graph_list=[]
    mapped = len(label_dict)
    label_dict[3] = mapped
        #c_sizes = [15] * 4
    for k in range(100,200):
        for j in range(10):
            g=nx.barabasi_albert_graph(k,4)
            graph_list.append(g)
            g_list.append(S2VGraph(g, 3, [0] * g.number_of_nodes()))
    save_graph_list(
                graph_list,
                os.path.join('saved_graphs', '{}_test.p'.format('barabasi')))
    graph_list=[]
    mapped = len(label_dict)
    label_dict[4] = mapped
        #c_sizes = [15] * 4
    for k in range(100,200):
        for j in range(10):
            c_sizes = np.random.choice(list(range(12, 17)), 2)
            g=n_community(c_sizes, p_inter=0.05)
            graph_list.append(g)
            g_list.append(S2VGraph(g, 4, [0] * sum(c_sizes)))
    save_graph_list(
                graph_list,
                os.path.join('saved_graphs', '{}_test.p'.format('community2')))
    graph_list=[]
    mapped = len(label_dict)
    label_dict[5] = mapped
        #c_sizes = [15] * 4
    for k in range(100,200):
        for i in range(10):
            g=nx.ladder_graph(k)
            graph_list.append(g)
            g_list.append(S2VGraph(g, 5, [0] * g.number_of_nodes()))
    save_graph_list(graph_list,
                    os.path.join('saved_graphs', '{}_test.p'.format('ladder')))
    graph_list=[]
    mapped = len(label_dict)
    label_dict[6] = mapped
        #c_sizes = [15] * 4
    for k in range(10,20):
        for j in range(10,20):
            for i in range(10):
                g=nx.grid_2d_graph(k, j)
                adj_matrix = nx.adjacency_matrix(g)
                g=nx.Graph(adj_matrix)
                graph_list.append(g)
                g_list.append(S2VGraph(g, 6, [0] * g.number_of_nodes()))
    save_graph_list(
                graph_list,
                os.path.join('saved_graphs', '{}_test.p'.format('grid')))
    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    if random :
        for g in g_list:
            g.node_features = torch.ones(len(g.node_tags), 1)
    else :
        for g in g_list:
            g.node_features = torch.zeros(len(g.node_tags), len(tagset))
            g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict), tag2index , len(tagset)

def load_graph_asS2Vgraph(graph_list,label,random,tag2index , lentagset):
    """Convert the nx.Graph list into a S2vGraph list ( preparing for GIN )"""
    g_list = []
    label_dict = {}
    feat_dict = {}
    mapped = label
    label_dict[label] = mapped
    for k in range(len(graph_list)):
        adj_matrix = nx.adjacency_matrix(graph_list[k])
        g=nx.Graph(adj_matrix)
        g_list.append(S2VGraph(g, label, [0] * graph_list[k].number_of_nodes()))
    
    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    for graph in g_list:
        graph.node_tags = list(dict(graph.g.degree).values())

    if random :
        for g in g_list:
            g.node_features = torch.ones(len(g.node_tags), 1)
    else :
        for g in g_list:
            g.node_features = torch.zeros(len(g.node_tags), lentagset)
            g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % lentagset)

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

def n_community(c_sizes, p_inter=0.01):
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = [G.subgraph(c) for c in nx.connected_components(G)]
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i + 1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
    # print('connected comp: ', len(list(nx.connected_component_subgraphs(G))))
    return G

# save a list of graphs
def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)
        
def pick_connected_component_new(G):
    # import pdb; pdb.set_trace()

    # adj_list = G.adjacency_list()
    # for id,adj in enumerate(adj_list):
    #     id_min = min(adj)
    #     if id<id_min and id>=1:
    #     # if id<id_min and id>=4:
    #         break
    # node_list = list(range(id)) # only include node prior than node "id"

    adj_dict = nx.to_dict_of_lists(G)
    for node_id in sorted(adj_dict.keys()):
        id_min = min(adj_dict[node_id])
        if node_id < id_min and node_id >= 1:
            # if node_id<id_min and node_id>=4:
            break
    node_list = list(
        range(node_id))  # only include node prior than node "node_id"

    G = G.subgraph(node_list)
    G = max(nx.connected_component_subgraphs(G), key=len)
    return G


def load_graph_list(fname, is_real=True):
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)
    return graph_list