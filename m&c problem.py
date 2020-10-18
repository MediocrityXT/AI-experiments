import sys
sys.setrecursionlimit(100)

def create(n):
    opList = []
    for i in range(1,n+1):
        for c in range(0,int(i/2)+1):
            opList.append((i-c,c))
        opList.append((0,i))
    return opList

def judge(s,operation,m,c):
    newS = s.copy()
    #judge operation is valid or not
    #and judge result is valid or not
    if(s[2]):
        #P operation
        if(operation[0]>newS[0] or operation[1]>newS[1]):
            valid = 0
            return [[],valid]
        else:
            newS[0] -= operation[0]
            newS[1] -= operation[1]
            newS[2] = 0
    else:
        #Q operation
        if(operation[0]>m-newS[0] or operation[1]>c-newS[1]):
            valid = 0
            return [[],valid]
        else:
            newS[0] += operation[0]
            newS[1] += operation[1]
            newS[2] = 1

    if((newS[0]>0 and newS[0]<newS[1]) or (m-newS[0]>0 and m-newS[0]<c-newS[1])):
        valid = 0
        return [[],valid]
    else:
        valid = 1
        return [newS,valid]

def test(list,s,m,c):
    global route
    global flag
    if(s==[0,0,0]):
        flag = True
        for i in route:
            print(i)
        print("步数:",len(route)-1)
        return

    for i in list:
        [newS,valid]=judge(s,i,m,c) # judge needs the boat and the ope
        #disallow to repeat the
        if(valid):
            if(route.count(newS)!=0):
                continue
            else:
                route.append(newS)
                test(list,newS,m,c)
        else:
            continue
    if(flag==False):
        route.pop()
    return

m=int(input('m='))
c=int(input('c='))
n=int(input('n=')) #number one boat can carry

s=[m,c,1]
route=[s]
flag = False
opList = create(n)
test(opList,s,m,c)
if(flag):
    print("可以安全渡河")
else:
    print("无法安全渡河！")

