import numpy as np
import networkx as nx
from random import randrange, choice
import copy
import operator
from sklearn.metrics import roc_auc_score
from munkres import Munkres

class Network:
    def __init__(self,G,connected=True,pctRemoval=0.01,clustering_after_last_node=True,node_removal_discrete=None,directed=False,DensityTolerance=0.05,ForbiddenList=None): #Initializer, determines the necessary values for our functions to run
        self.Greal = G
        self.num_nodes = self.Greal.number_of_nodes() #Gets number of nodes in the graph
        self.guarantee_connected = connected
        if node_removal_discrete != None:
            self.num_nodes_to_remove = node_removal_discrete
            self.nodes_to_predict = self.num_nodes_to_remove
        else:
            self.num_nodes_to_remove = int(np.ceil(self.num_nodes*pctRemoval)) #Example 198 node network, with 0.01 pctRemoval gets 1.98 which is rounded to 2.00 and then turned into an integer 2 
            self.nodes_to_predict = self.num_nodes_to_remove #probably ephemeral, mostly done for ease-of-reading
        self.TestGraphEditDistance = 0
        if clustering_after_last_node:
            self.Qcontrol = 1
        else:
            self.Qcontrol = 0
        self.directed = directed
        self.ComparisonList = []
        self.VerificationList = []
        self.VerificationListMirror = []
        self.bool_list = []
        self.DensityTolerance = DensityTolerance
        self.ClusteringExclusionIndexList = []
        self.SafeRemovalList = list(np.setdiff1d(list(self.Greal),ForbiddenList))

    def create_test_network(self): #This function will replicate Greal into Gtest and subsequently remove nodes from Gtest
        Gtest = copy.deepcopy(self.Greal) #Deep Copy is necessary here to prevent Gtest from being a reference to Greal, we need Greal fully intact so that we can accurately use it for comparison later while evaluation the prediction's effectiveness
        K=0 #Control variable for a loop
        F=0 #Control variable to prevent endless loops
        if self.guarantee_connected: #In cases where we want to maintain fully connected graphs
            while K < self.num_nodes_to_remove: #We will run this loops until we've successfuly removed all the nodes we are supposed to
                if F == 25: #25 is an arbitrary stopping point, easy to convert to one that scales with network size.
                    raise Exception('Network not well suited to keep fully connected')      
                i = choice(self.SafeRemovalList) #to prevent bias we select a node index entirely at random
                if Network.test_connected(self,Gtest,i):
                    Gtest.remove_node(i)
                    self.SafeRemovalList.remove(i)
                    K+=1 #We successfully removed a node so we iterate K
                else:
                    self.SafeRemovalList.remove(i)
                    F+=1
                    continue          
        else: #If we don't care about the graph being fully connected ; the procedure is exactly the same as above, just with no checking if the node we're removing is problematic or not
            while K < self.num_nodes_to_remove:
                i = randrange(0,stop=(Gtest.number_of_nodes()),step=1)
                Gtest.remove_node(list(Gtest)[i])
                self.ComparisonList.append(list(Gtest.neighbors(list(Gtest)[i])))
                K+=1
        self.TestGraphEditDistance = self.Greal.number_of_edges() - Gtest.number_of_edges()
        return Gtest

    def create_boolmatrix(self):
        for i in range(int(len(self.ComparisonList))): #ComparisonList is a list of tuples, i runs through every tuple
            for j in range(int(len(self.ComparisonList[i]))): # j runs through every element of each tuple
                for k in range(j+1,len(self.ComparisonList[i])): # k runs 'ahead' of j to avoid (j,j) or (j,j-1) elements
                    self.VerificationList.append([self.ComparisonList[i][j],self.ComparisonList[i][k]])
                    self.VerificationListMirror.append([self.ComparisonList[i][k],self.ComparisonList[i][j]]) #We don't know the order of predicted additions, might be (i,j) or (j,i) so the safe approach is to recreate a mirrored list and compare to both possibilities

    def test_connected(self,Gtest,i):
        edges_of_i = Gtest.neighbors(i) #We select all the edges connecting to the selected node to verify if it's a safe node to remove
        list_of_edges_of_i = list(edges_of_i)
        if (len(list_of_edges_of_i) > 1):
            for k in list_of_edges_of_i: #remove all (i,k) edges
                Gtest.remove_edge(i,k)
            if nx.number_connected_components(Gtest)==2: #number of components needs to be 2 for the node to be safe to remove, if it's >2 then it means we have (#Components - 1) subgraphs and the isolated node which means we've split the graph
                self.ComparisonList.append(list_of_edges_of_i)
                return True
            else:
                for j in list_of_edges_of_i:
                    Gtest.add_edge(i,j)
                return False

    def create_phantoms(self,Gtest):
        Gphantom = copy.deepcopy(Gtest)
        number_of_nodes = Gphantom.number_of_nodes()
        for i in range(number_of_nodes):
            Gphantom.add_node('DuplicatedNode'+str(list(Gphantom)[i])) #Doubles Network size, each new node is named 'DuplicatedNode' + the original label
        for j in range(number_of_nodes):
            neighbors_of_j = nx.all_neighbors(Gphantom,list(Gphantom)[j]) #Grabs all neighbors of node j
            neighbors_of_j = [item for item in neighbors_of_j if (('DuplicatedNode' not in item))] #purges all Duplicated Nodes from the neighbor list
            for k in list(neighbors_of_j): #run for every neighbour of j
                Gphantom.add_edge(str('DuplicatedNode'+str(list(Gphantom)[j])),k) #Connect Node j's duplicate to j's neighbors
            Gphantom.add_edge(str('DuplicatedNode'+str(list(Gphantom)[j])),list(Gphantom)[j]) #Connect Node j to j's duplicate
        return Gphantom

    def check_affinity(self,Gphantom,AffinityType='AA'):
        ScoreList=[]
        if AffinityType == 'AA':
            Affinity = nx.adamic_adar_index(Gphantom)
        elif AffinityType == 'RA':
            Affinity = nx.resource_allocation_index(Gphantom)
        elif AffinityType == 'JC':
            Affinity = nx.jaccard_coefficient(Gphantom)
        elif AffinityType == 'PA':
            Affinity = nx.preferential_attachment(Gphantom)
        else:
            raise Exception('Please select AA, for Adamic-Adar index, RA, for Resource Allocation index, PA, for Preferential Attachment index, or JC for Jaccards Coefficient index.')
        for u,v,p in Affinity: # u,v nodes, p score
            ScoreList.append([u,v,p])
        ScoreList = [item for item in ScoreList if (('DuplicatedNode' in item[0]) and ('DuplicatedNode' in item[1]))] #Purge all scores not related to duplicated nodes
        ScoreList.sort(key=operator.itemgetter(2),reverse=True) #Sort the list by decreasing scores
        self.AUCscores = [c for a,b,c in ScoreList] #Pick out the sorted score list for AUC calculation
        self.ClusteringList = [(Network.string_subtract(self,a,'DuplicatedNode'),Network.string_subtract(self,b,'DuplicatedNode')) for a,b,c in ScoreList] #Clean up the list to make it easier to match up things later. The list should have only the original node labels

    def create_verification_list(self):
        #VerificationList = [(Network.string_subtract(self,a,'DuplicatedNode'),Network.string_subtract(self,b,'DuplicatedNode')) for a,b,c in self.ScoreList] #Clean up the labels from Duplication
        VerificationList = self.ClusteringList
        VerificationList = [[a,b] for a,b in VerificationList] #Turn elements into tuples to match up with earlier data organisation so comparisons can be made
        for i in range(len(VerificationList)):
            if VerificationList[i] in self.VerificationList or VerificationList[i] in self.VerificationListMirror: #Check if the clustering pairs are the pairs we removed earlier to calculate AUC
                self.bool_list.append(1)
            else:
                self.bool_list.append(0)

    def string_subtract(self,a,b): #Example: a='MaximalCat', b='Maximal', c = string_subtract(a,b) = 'Cat'
        return "".join(a.rsplit(b))

    def add_predicted_nodes(self,Gtest):
        self.Gexpanded = copy.deepcopy(Gtest)
        self.PhantomIDMatchingList = []
        self.CreatedIDMatchingList = []
        K = 0
        Q = 0
        while Q < self.nodes_to_predict+self.Qcontrol: #Qcontrol = 0 if no clustering after last node, Q=1 for clustering until self.nodes_to_predict is achieved
            while Q < self.nodes_to_predict:
                if self.ClusteringList[K][0] in self.PhantomIDMatchingList or self.ClusteringList[K][1] in self.PhantomIDMatchingList: #Check if we already used either of the elements in the tuple
                    Network.cluster_virtual_nodes(self,Q,K)
                    if not Network.check_density(self,Gtest):
                        Network.force_node(self,Q,K)
                        Q+=1
                else: #If we need to create a node, always fires before the if
                    Network.add_node(self,Q,K)
                    Q+=1 #Iterate Q every time we add a node
                K+=1 #Iterate K every time we use self.ClusteringList
                if Network.clustering_failure_condition_check(self,K):
                    break
            if self.ClusteringList[K][0] in self.PhantomIDMatchingList or self.ClusteringList[K][1] in self.PhantomIDMatchingList:
                Network.cluster_virtual_nodes(self,Q,K)
                if not Network.check_density(self,Gtest):
                    #Network.force_node(self,Q,K)
                    Q+=1
            else:
                Q+=1
            K+=1
            if Network.clustering_failure_condition_check(self,K):
                break
            if K in self.ClusteringExclusionIndexList:
                K+=1
        delattr(self,'ClusteringList')

    def force_node(self,Q,K):
        for R in range(K,len(self.ClusteringList)):
            if self.ClusteringList[R][0] not in self.PhantomIDMatchingList and self.ClusteringList[R][1] not in self.PhantomIDMatchingList:
                Network.add_node(self,Q,R)
                self.ClusteringExclusionIndexList.append(R)
                break
            if R == len(self.ClusteringList):
                raise Exception('force_node can\'t find a node to cluster')

    def clustering_failure_condition_check(self,K):
        if K >= len(self.ClusteringList):
            print('Failure')
            return True
        else:
            return False
    
    def cluster_virtual_nodes(self,Q,K):
        if self.ClusteringList[K][0] in self.PhantomIDMatchingList: #If it's the first element of the tuple
            Node = self.CreatedIDMatchingList[int(self.PhantomIDMatchingList.index(self.ClusteringList[K][0])/2)] #Check which created node it corresponds to
            self.Gexpanded.add_edge(Node,self.ClusteringList[K][1])
        else: #If it's the second
            Node = self.CreatedIDMatchingList[int(self.PhantomIDMatchingList.index(self.ClusteringList[K][1])/2)]
            self.Gexpanded.add_edge(Node,self.ClusteringList[K][0])
        self.PhantomIDMatchingList.extend([self.ClusteringList[K][0],self.ClusteringList[K][1]]) #Extend is used so that 2 indices are added rather than a tuple, using append here will create something like A=[1 2 3], A.append([4,5]) -> A=[1 2 3 [4 5]] where A(1:3) are single elements and A(4) = [4 5]. DO NOT USE APPEND HERE.
        self.CreatedIDMatchingList.append(Node)
    
    def add_node(self,Q,K):
        self.Gexpanded.add_node('PredictedNode'+str(Q)) #Create a node
        self.Gexpanded.add_edge('PredictedNode'+str(Q),self.ClusteringList[K][1]) #Connect it to the predicted neighbor(s)
        self.Gexpanded.add_edge('PredictedNode'+str(Q),self.ClusteringList[K][0]) #Second neighbor connection
        self.PhantomIDMatchingList.extend([self.ClusteringList[K][0],self.ClusteringList[K][1]]) #Bookkeeping
        self.CreatedIDMatchingList.append('PredictedNode'+str(Q)) #Bookkeeping  

    def check_density(self,Gbase):
        GraphDensityBase = 2*Gbase.number_of_edges()/Gbase.number_of_nodes()
        GraphDensityExpansion = 2*self.Gexpanded.number_of_edges()/self.Gexpanded.number_of_nodes()
        if GraphDensityExpansion > GraphDensityBase*(1+self.DensityTolerance):
            return False
        else:
            return True       

    def auc_score(self): #AUC needs two arrays of equal size. One with the scores and another that validates the associated score as true or false.
        # y_true=np.zeros(len(self.bool_list),dtype=int)
        # for K in range(len(self.bool_list)): #Could probably avoid this loop if I defined self.bool_list as 0s and 1s to begin with
        #     if self.bool_list[K]:
        #         y_true[K]=1
        #     else:
        #         y_true[K]=0
        y_scores=np.array(self.AUCscores) #Numpy is a sweetheart that converts lists to arrays without much fuss
        y_true=np.array(self.bool_list)
        auc_score=roc_auc_score(y_true,y_scores)
        delattr(self,'bool_list')
        delattr(self,'VerificationList')
        delattr(self,'VerificationListMirror')
        return auc_score

    def evaluation(self):
        PredictedComparisonList = []
        number_of_elements_i = (self.Gexpanded.number_of_nodes()-self.num_nodes_to_remove)
        number_of_elements_f = self.Gexpanded.number_of_nodes()
        for K in range(number_of_elements_i,number_of_elements_f):
            PredictedComparisonList.append(list(self.Gexpanded.neighbors(list(self.Gexpanded)[K])))
        #GEDcostMatrix = np.matlib.zeros(((self.num_nodes_to_remove),(self.num_nodes_to_remove)))
        GEDcostMatrix = []
        for i in range(self.num_nodes_to_remove):
            GEDcostMatrix.append([])
            for _ in range(self.num_nodes_to_remove):
                GEDcostMatrix[i].append(0)
        for R in range(0,self.num_nodes_to_remove):
            for P in range(0,self.num_nodes_to_remove):
                if len(PredictedComparisonList[P])-len(self.ComparisonList[R]) < 0:
                    mismatchedElements_P_notin_R = np.setdiff1d(PredictedComparisonList[P],self.ComparisonList[R])
                    GEDcostMatrix[R][P] = abs(len(PredictedComparisonList[P])-len(self.ComparisonList[R])) + len(mismatchedElements_P_notin_R)
                elif len(PredictedComparisonList[P])-len(self.ComparisonList[R]) == 0:
                    mismatchedElements_P_notin_R = np.setdiff1d(PredictedComparisonList[P],self.ComparisonList[R])
                    GEDcostMatrix[R][P] = len(mismatchedElements_P_notin_R)
                else: #len(PredictedComparisonList[P])-len(self.ComparisonList[R]) > 0:
                    mismatchedElements_P_notin_R = np.setdiff1d(PredictedComparisonList[P],self.ComparisonList[R])
                    GEDcostMatrix[R][P] = len(mismatchedElements_P_notin_R) + abs(len(PredictedComparisonList[P])-len(self.ComparisonList[R]))
        delattr(self,'ComparisonList')
        if self.num_nodes_to_remove == 1:
            totalcost = GEDcostMatrix[0][0]
        elif self.num_nodes_to_remove == 2:
            if (GEDcostMatrix[0][0]+GEDcostMatrix[1][1]) > (GEDcostMatrix[1][0] + GEDcostMatrix[0][1]):
                totalcost = GEDcostMatrix[1][0] + GEDcostMatrix[0][1]
            else:
                totalcost = GEDcostMatrix[0][0] + GEDcostMatrix[1][1]
        else: #self.num_nodes_to_remove > 2:
            #CostMatrix = Munkres(GEDcostMatrix,self.num_nodes_to_remove)
            #totalcost = Munkres.run_munkres(CostMatrix)
            m = Munkres()
            indexes = m.compute(GEDcostMatrix)
            totalcost = 0
            for row, column in indexes:
                value = GEDcostMatrix[row][column]
                totalcost += value
        return totalcost

    def comparison(self,GraphEditDistance): #it's a ratio
        Ratio = GraphEditDistance / self.TestGraphEditDistance
        return Ratio