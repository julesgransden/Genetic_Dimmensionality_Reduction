import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random

df = pd.read_csv("vectorizedData.csv")

class Generation():
    
    def __init__(self, population, sln):
        self.population = population
        self.sln = sln
    
    #returns sorted population of all solutions from highest score to lowest
    def sortPop(self):
        return dict(sorted(self.population.items(),key=lambda x:x[1], reverse=True))
        
    #Select parents for mutation and child forming
    #we will pick the top X performering solutions
    #we will match them in order of performance(ex: (1,2), (2,3), ... (n, n+1))
    #returns a list o tuples representing the sln 
    def ParentSelection1(self, x):
        parents =list(self.sortPop())
        print(parents)
        parents = parents[:x]
        i=0
        res = []
        while i<x-1:
            res.append((parents[i],parents[i+1]))
            i+=1
        return res
    
    #similar to ParentSelection1, but uses the top performing solution as parent for all new children
    #ex; (1,2),(1,3),...(1,n) 
    def ParentSelection2(self,x):
        parents = self.sortPop()
        parents = parents[:x]
        i=0
        res = []
        while i<x-1:
            res.append((0,i+1))
            i+=1
        return res

    #Return next generation of children,array of genes
    def ProducenewGen(self, parents):
        newsln = []
        for m,d in parents:
            geneM = self.sln[m]
            geneD = self.sln[d]
            #perform single point crossover
            #child1,child2 = MultiplePointCrossoverFct(geneM,geneD, [250,500,750,1000])
            child1,child2 = singlePointCrossoverFct(geneM,geneD, 750)
            child1 = mutation(child1)
            child2 = mutation(child2)
            newsln.extend([list(child1),list(child2)])
        topPerf1,topPerf2 = list(self.sortPop())[0],list(self.sortPop())[1]
        newsln.extend([self.sln[topPerf1],self.sln[topPerf2]])
        return newsln
    
    

def MultiplePointCrossoverFct(gene1,gene2,points):
    for x in points:
        newA = np.append(gene1[:x], gene2[x:])
        newB = np.append(gene2[:x], gene1[x:])
    return newA, newB

def singlePointCrossoverFct(gene1,gene2,x):
    newA = np.append(gene1[:x], gene2[x:])
    newB = np.append(gene2[:x], gene1[x:])
    return newA, newB

#genetic mutation on children genes 
#modify randomly 10 binary values in gene
#return new gene
def mutation(gene):
    for _ in range(10):
        randomIndex = random.randint(0, len(gene)-1)  # Fixed indexing
        gene[randomIndex] = 1 - gene[randomIndex]  # Flipping 0 to 1 and vice versa
    return gene

def splitData(em,tf=0.75): #tf = training fraction & em=embbeded dataframe
    X = em.iloc[:,1:]
    Xtrain = X[:int(tf*X.shape[0])]
    Xtest = X[int(tf*X.shape[0]):] 
    Y = em.iloc[:,1]
    Ytrain = Y[:int(tf*Y.shape[0])]
    Ytest = Y[int(tf*Y.shape[0]):] 
    return Xtrain, Xtest, Ytrain, Ytest

#create genetic representation of sln, binary representation
def generateGeneticSLN(df):
    l = df.shape[1]-1
    gene = []
    for i in range(l):
        gene.append(random.randint(0,1))
    return gene

def generateDFSolution(gene,X_train, X_test):
    sln_train= pd.DataFrame()
    sln_test= pd.DataFrame()
    for i in range(1,len(gene)):
        curr = gene[i]
        if curr!=0:
            #select feature column and add in solution df
            tr = X_train.iloc[:,i]
            te = X_test.iloc[:,i]
            sln_train = pd.concat([sln_train, tr], axis=1)
            sln_test = pd.concat([sln_test, te], axis=1)
            
    return sln_test, sln_train

def genHeuristicValue(gene,X_train, X_test,Y_train,Y_Test):
    sln_test, sln_train = generateDFSolution(gene,X_train, X_test)
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(sln_train, Y_train)
    #testing set and get accuracy
    y_pred = clf.predict(sln_test)
    return accuracy_score(Y_Test, y_pred)

def initialPopulation(df, size):
    Xtrain, Xtest, Ytrain, Ytest = splitData(df)
    #we will generate a dictionary where each key value pair represents a sln and its heuristic value 
    #initialize our genetic algorithm
    population = { }
    sln = []
    for i in range(size):
        sln.append(generateGeneticSLN(df))
        population[i] = genHeuristicValue(sln[i],Xtrain, Xtest, Ytrain, Ytest)
        print(population)
    return population, sln

#perform genetic feature selection
def geneticFeatureSelection(df, initPop_size, nbGens):
    Xtrain, Xtest, Ytrain, Ytest = splitData(df)
    
    population,sln = initialPopulation(df, initPop_size)
    #define initial generation
    gen0 = Generation(population,sln)
    tree = [gen0]
    for i in range(nbGens):
        gen = tree[i]
        parents = gen.ParentSelection1(int(0.5*len(gen.sln)))
        newSln = gen.ProducenewGen(parents)
        newpop = {}
        Xtrain, Xtest, Ytrain, Ytest = splitData(df)
        for i in range(len(newSln)):
                newpop[i] = genHeuristicValue(newSln[i],Xtrain, Xtest, Ytrain, Ytest)
                print(newpop)
        newgen = Generation(newpop,newSln)
        tree.append(newgen)
    return tree[-1].population
                


if __name__ == "__main__":
    
    population,sln = initialPopulation(df, 10)
    gen0 = Generation(population,sln)
    
    Xtrain, Xtest, Ytrain, Ytest = splitData(df)
    
    tree = [gen0]
    for i in range(5):
        gen = tree[i]
        parents = gen.ParentSelection1(int(0.5*len(gen.sln)))
        newSln = gen.ProducenewGen(parents)
        newpop = {}
        Xtrain, Xtest, Ytrain, Ytest = splitData(df)
        for i in range(len(newSln)):
                newpop[i] = genHeuristicValue(newSln[i],Xtrain, Xtest, Ytrain, Ytest)
                print(newpop)
        newgen = Generation(newpop,newSln)
        tree.append(newgen)