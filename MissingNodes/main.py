import numpy as np
import networkx as nx
import copy
from supportfunctions import create_forbidden_list, frange, parse_G
from network import Network
#Functionality Definitions
np.set_printoptions(threshold=np.inf)
#Input
connected = True
repetitions = 50
#filename = 'netscience'
labeltype='labels'
MaxNodesRemoved=160
AffinityType='AA'
#File Parsing
#Aux Code
for filename in ['dolphins','lesmis','adjnoun','polbooks']:
    G = nx.read_gml(filename+'.gml',label=labeltype)
    G = parse_G(G,labeltype,connected)
    ForbiddenList = create_forbidden_list(G)
    for DensityTolerance in frange(0.005,0.50,0.005):
        F=1
        AUC=np.zeros(repetitions,dtype=float)
        Ratio=np.zeros(repetitions,dtype=float)
        for i in range(0, repetitions):
            #Network Object Creation & Parsing
            A = Network(G,connected=connected,node_removal_discrete=3,ForbiddenList=ForbiddenList,DensityTolerance=DensityTolerance)
            if i == 0:
                Nodes_Removed = A.nodes_to_predict
            try:
                Gtest = Network.create_test_network(A)
            except:
                F=0
                break
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
        if F > 0:
            PrintTolerance=DensityTolerance*100
            np.savetxt(filename+'_'+'AA'+'_AUC'+'_'+str(PrintTolerance)+'.csv',AUC,delimiter=',')
            np.savetxt(filename+'_'+'AA'+'_Ratio'+'_'+str(PrintTolerance)+'.csv',Ratio,delimiter=',')
        else:
            pass