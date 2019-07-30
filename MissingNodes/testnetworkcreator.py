import numpy as np
import networkx as nx
import os
import random
from supportfunctions import parse_G

filename = 'email.gml'
G = nx.read_gml(filename)
labeltype = 'id'

G = parse_G(G,labeltype,True)

removal_rates = np.arange(0.05,0.25,0.025)
nodes_to_remove = np.ceil(G.number_of_nodes()*removal_rates)

for number in nodes_to_remove:
    while True:
        ExclusionList=[]
        EdgesToRecover=[]
        F = G.copy()
        i=0
        while i < number:
            random_choice = random.choice(list(F))
            while random_choice in ExclusionList:
                random_choice = random.choice(list(F))
            if len(list(F.neighbors(random_choice))) < 2:
                ExclusionList.append(random_choice)
            else:
                EdgesToRecover.append(list(F.neighbors(random_choice)))
                F.remove_node(random_choice)
                i+=1
        if nx.is_connected(F):
            filetowrite ='email_'+str(number)+'_removed'
            nx.write_gml(F,filetowrite+'.gml')
            with open(filetowrite+".txt", "w") as f:
                for s in EdgesToRecover:
                    f.write(str(s) +"\n")
            break