class node:
    s=[]
    g=0
    h=0
    last=None
    def __init__(self,s1,g1,h1,last1):
        self.s=s1
        self.g=g1
        self.h=h1
        self.last=last1

def display(s):
    for i in range(0,3):
        print(s[i*3]," ",s[i*3+1]," ",s[i*3+2])
    print("\n")

def getH(s,sg):
    h=0
    for i in range(9):
        if(s[i]==' '):
            continue
        j=sg.index(s[i])
        h += abs(i%3 - j%3)+abs(i//3 - j//3)
    return h

def up(os):
    s=os.copy()
    index = s.index(' ')
    if(index - 3 >= 0):
        temp = s[index]
        s[index] = s[index - 3]
        s[index - 3] = temp
        return s
    else:
        return
def down(os):
    s=os.copy()
    index = s.index(' ')
    if(index + 3 < 9):
        temp = s[index]
        s[index] = s[index + 3]
        s[index + 3] = temp
        return s
    else:
        return
def left(os):
    s=os.copy()
    index = s.index(' ')
    if((index - 1)//3 == index // 3):
        temp = s[index]
        s[index] = s[index - 1]
        s[index - 1] = temp
        return s
    else:
        return
def right(os):
    s=os.copy()
    index = s.index(' ')
    if((index + 1)//3 == index // 3):
        temp = s[index]
        s[index] = s[index + 1]
        s[index + 1] = temp
        return s
    else:
        return

def checkExist(x,list):
    for index in range(len(list)):
        if x == list[index].s:
            return [True,index]
    return [False,-1]

def spread(s):
    res = []
    if(up(s)!=None):
        res.append(up(s))
    if(down(s)!=None):
        res.append(down(s))
    if(left(s)!=None):
        res.append(left(s))
    if(right(s)!=None):
        res.append(right(s))
    return res
def show(sgNode):
    if(sgNode.last!=None):
        display(sgNode.s)
        show(sgNode.last)
    else:
        display(sgNode.s)

def compareF(elem):
    return elem.g+elem.h

def Astar(sg):
    flag=False
    global open
    global closed
    while(len(open)):
        bestnode = open[0]
        open.pop(0)
        closed.append(bestnode)
        if(bestnode.s == sg):
            flag = True
            show(bestnode)
            break
        else:
            for i in spread(bestnode.s):
                sucNode = node(i,bestnode.g+1,getH(i,sg),bestnode)
                if(checkExist(i,open)[0]==False):
                    if(checkExist(i,closed)[0]==False):
                        open.append(sucNode)
                    else:
                        oldIndex = checkExist(i,closed)[1]
                        if(sucNode.g < oldIndex):
                            #replace the old node
                            closed.pop(oldIndex)
                            open.append(sucNode)
                else:
                    oldIndex = checkExist(i,open)[1]
                    if(sucNode.g < oldIndex):
                        #replace the old node
                        open.pop(oldIndex)
                        open.insert(oldIndex,sucNode)
            open.sort(key=compareF)
    if(flag):
        print("success")
    else:
        print("failure")


s=[2,8,3,1,' ',4,7,6,5]
sg=[1,2,3,8,' ',4,7,6,5]

n1=node(s,0,getH(s,sg),None)
open=[n1]
closed=[]
Astar(sg)