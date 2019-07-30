import networkx as nx
import numpy as np
import copy

#Functions

def test_connected(Gtest,i):
    edges_of_i = Gtest.neighbors(i) #We select all the edges connecting to the selected node to verify if it's a safe node to remove
    list_of_edges_of_i = list(edges_of_i)
    if (len(list_of_edges_of_i) > 1):
        for k in list_of_edges_of_i: #remove all (i,k) edges
            Gtest.remove_edge(i,k)
        if nx.number_connected_components(Gtest)==2: #number of components needs to be 2 for the node to be safe to remove, if it's >2 then it means we have (#Components - 1) subgraphs and the isolated node which means we've split the graph
            for j in list_of_edges_of_i:
                Gtest.add_edge(i,j)
            return True
        else:
            for j in list_of_edges_of_i:
                Gtest.add_edge(i,j)
            return False

def parse_G(G,labeltype,connected):
    if labeltype=='id':
        ListOfUtterStupidity=[]
        for i in range(G.number_of_nodes()):
            ListOfUtterStupidity.append('Label'+str(list(G)[i]))
        mapping = dict(zip(G, ListOfUtterStupidity))
        G = nx.relabel_nodes(G, mapping)
    if connected:
        if nx.is_connected(G):
            pass
        else:
            G = G.subgraph(max(nx.connected_components(G),key=len)).copy()
    return G

def create_forbidden_list(G):
    ForbiddenList=[]
    #IsolatedList=[]
    for item in list(G):
        if not test_connected(G,item):
            ForbiddenList.append(item)
    return ForbiddenList

def frange(start,stop,step):
    i = start
    while i < stop:
        yield i
        i += step