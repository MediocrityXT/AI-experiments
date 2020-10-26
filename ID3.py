from os import system, write
import numpy as np
import os
import math

def loadTrainData():
    with open('AI experiments/data/traindata.txt', 'r') as f:
        traindata=f.read()
    x=traindata.split('[\n',2)[1].split("\n]",2)[0] #remove first and last row of file
    with open('TempTrainData.txt','w') as f:
        f.write(x)
    data=np.loadtxt('TempTrainData.txt')
    os.remove('TempTrainData.txt')
    return data
def loadTestData():
    with open('AI experiments/data/testdata.txt', 'r') as f:
        traindata=f.read()
    x=traindata.split('[\n',2)[1].split("\n]",2)[0] #remove first and last row of file
    with open('TempTestData.txt','w') as f:
        f.write(x)
    data=np.loadtxt('TempTestData.txt')
    os.remove('TempTestData.txt')
    return data

def Info(l):
    res=[0,0,0]
    for i in [1,2,3]:
        if(l.count(i)!=len(l) and l.count(i)!=0):
            res[i-1]=l.count(i)/len(l)*math.log2(l.count(i)/len(l))
        else:
            res[i-1]=0
    info=-res[0]-res[1]-res[2]
    return info
def InfoA(newL,mean):
    frontL=[]
    rearL=[]
    #divide the list not by i but by the mean
    for j in range(len(newL)):
        if(newL[j][0]<=mean):
            frontL.append(newL[j][1])
        else:
            rearL.append(newL[j][1])
    return [(len(frontL)*Info(frontL)+len(rearL)*Info(rearL))/len(newL),len(frontL)]
def SplitPoint(c):
    newL=sorted(c,key=lambda x:x[0])
    #print("总数据量：",len(newL))
    res=[]
    for i in range(len(newL)-1):
        mean=(newL[i][0]+newL[i+1][0])/2
        temp=InfoA(newL,mean)
        res.append([mean,temp[0],temp[1]])
    res.sort(key=lambda x:x[1])
    #print('best division point is: ',res[0][0])
    return res[0]

def SplitPointInAllAttr(data):
    res=[0,0,0,0]
    l=list(data[:,4])
    
    print("Info at begining is:",Info(l))
    c1=[]
    c2=[]
    c3=[]
    c4=[]
    for index,i in enumerate(data[:,0]):
        c1.append([i,l[index]])
    for index,i in enumerate(data[:,1]):
        c2.append([i,l[index]])
    for index,i in enumerate(data[:,2]):
        c3.append([i,l[index]])
    for index,i in enumerate(data[:,3]):
        c4.append([i,l[index]])
    
    res[0]=SplitPoint(c1)+["c1"]
    res[1]=SplitPoint(c2)+["c2"]
    res[2]=SplitPoint(c3)+["c3"]
    res[3]=SplitPoint(c4)+["c4"]
    res.sort(key=lambda x:x[1])

    return res[0]

def isSameClass(data):
    l=list(data[:,4])
    if(l.count(l[0])==len(l)):
        return True
    else:
        return False
def Classify(row,root) -> int:
    while(root.isPure==0):
        Class=int(root.attr[1])
        if(row[Class-1]<=root.point):
            root=root.lchild
        else:
            root=root.rchild
    return int(root.attr[1])
def test(data,root):
    total = len(data)
    rightCnt=0
    for index in range(len(data[:,4])):
        row=list(data[index,:])
        if(Classify(row,root)==row[4]): 
            rightCnt+=1
    print("test accuracy:"+ str(rightCnt/total))
    
class node:
    def __init__(self,data,attr=0,point=0,father=0,lchild=0,rchild=0) -> None:
        self.data=data
        self.attr=attr
        self.point=point
        self.father=father
        self.lchild=lchild
        self.rchild=rchild
        self.isPure=0
                
    def doSplit(self):
        data=self.data
        if(isSameClass(data)):
            self.attr="c"+str(data[0,4])[0]
            self.isPure=1
            return
        else:
            pointTuple=SplitPointInAllAttr(data)
            #pointTuple is [mean,infoA,leftNum,attr]
            attr=int(pointTuple[3][1])
            self.attr=pointTuple[3]
            self.point=pointTuple[0]
            
            #split left and right
            left=np.zeros((pointTuple[2],5))
            right=np.zeros((len(data)-pointTuple[2],5))
            leftCnt=0
            rightCnt=0
            for i in range(len(data[:,0])):
                if(data[i,attr-1] <= pointTuple[0]):
                    left[leftCnt]=data[i]
                    leftCnt+=1
                else:
                    right[rightCnt]=data[i]
                    rightCnt+=1
            
            leftNode = node(left,father=self)
            rightNode = node(right,father=self)
            self.lchild=leftNode
            self.rchild=rightNode

            leftNode.doSplit()
            rightNode.doSplit()
    def displayDividePoint(self):
        if(self.isPure):
            return ":纯净节点"
        else:
            return ":"+str(self.point)
    def display(self):
        if(self.father==0):
            res =  str(self.attr)+" : "+str(self.point)+" 根节点" +"--左子节点为"+str(self.lchild.attr)+self.lchild.displayDividePoint()+"--右子节点为"+str(self.rchild.attr)+self.rchild.displayDividePoint()
        else:
            if(self.isPure==0):
                res = str(self.attr)+" : "+str(self.point)+"--左子节点为"+str(self.lchild.attr)+self.lchild.displayDividePoint()+"--右子节点为"+str(self.rchild.attr)+self.rchild.displayDividePoint()+"----父节点为"+str(self.father.attr)+":"+str(self.father.point)
            else:
                res = str(self.attr)+" : 纯净节点 "+"----父节点为"+str(self.father.attr)+":"+str(self.father.point)
        print(res)
        if(self.lchild!=0):
            self.lchild.display()
        if(self.rchild!=0):
            self.rchild.display()



data=loadTrainData()
print(data)
rootNode = node(data)
rootNode.doSplit()
rootNode.display()

testData=loadTestData()
test(testData,rootNode)


