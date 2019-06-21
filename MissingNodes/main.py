import numpy as np
import networkx as nx
import copy
from supportfunctions import create_forbidden_list, frange, parse_G
from network import Network
#Functionality Definitions
np.set_printoptions(threshold=np.inf)
#Input
connected = True
repetitions = 10
filename = 'dolphins'
labeltype='labels'
MaxNodesRemoved=5
AffinityType='AA'
DensityTolerance=0.005
#File Parsing
G = nx.read_gml(filename+'.gml',label=labeltype)
G = parse_G(G,labeltype,connected)
ForbiddenList = create_forbidden_list(G)
#Aux Code
for node_removal_discrete in range(3,MaxNodesRemoved):
    F=1
    AUC=np.zeros(repetitions,dtype=float)
    Ratio=np.zeros(repetitions,dtype=float)
    for i in range(0, repetitions):
        #Network Object Creation & Parsing
        A = Network(G,connected=connected,node_removal_discrete=node_removal_discrete,ForbiddenList=ForbiddenList,DensityTolerance=DensityTolerance)
        if i == 0:
            Nodes_Removed = A.nodes_to_predict
        try:
            Gtest = Network.create_test_network(A)
            #AUC
            Network.create_boolmatrix(A)
            Gphantom = Network.create_phantoms(A,Gtest)
            Network.check_affinity(A,Gphantom,AffinityType=AffinityType)
            del Gphantom
            Network.create_verification_list(A)
            AUC[i] = Network.auc_score(A)
            print('AUC = ',AUC[i])
            Network.add_predicted_nodes(A,Gtest)
            GED = Network.evaluation(A)
            Ratio[i] = GED/A.TestGraphEditDistance
            print('Ratio = ', Ratio[i])
            del A, Gtest
        except:
            F=0
            break
    if F > 0:
        np.savetxt(filename+'_'+'AA'+'_AUC'+'_'+str(node_removal_discrete)+'.csv',AUC,delimiter=',')
        np.savetxt(filename+'_'+'AA'+'_Ratio'+'_'+str(node_removal_discrete)+'.csv',Ratio,delimiter=',')
    else:
        pass