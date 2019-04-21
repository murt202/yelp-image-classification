import pandas as p
import numpy as np
import math as mt

class Node:
    def __init__(self,cols=-1,vals=-1,rbr=None,lbr=None,mov=[]):
        self.column=cols
        self.value=vals
        self.rightbranch=rbr
        self.leftbranch=lbr
        self.images=mov

def uniqueclasscount(rows):
    count = defaultdict(lambda: 0)
    for i in rows:
        r = i[len(i)-1]
        count[r]+=1
    return dict(count)

def entropy(rows):
    count=uniqueclasscount(rows)

    val=0
    for i in count.keys():
        p=count[i]/float(rows.shape[0])
        val-=p*mt.log2(p)

    return val
def split(rows,col,val):
    sr = np.zeros(shape=[0, rows.shape[1]])
    sl = np.zeros(shape=[0, rows.shape[1]])
    x= np.zeros(shape=[1,rows.shape[1]])
    for i in range(rows.shape[0]):
        x = rows[i]
        x = x.reshape(1, -1)
        if(rows[i][col]>=val):
            sr=np.append(sr,x,axis=0)
        else:
            sl = np.append(sl,x,axis=0)
    return sr,sl

def printtree(tree,indent=''):

    if len(tree.images)!=0:
        print (tree.images)
    else:

        print ('Column ' + str(tree.column)+' : '+str(tree.value)+'? ')


        print (indent+'Right->',)
        printtree(tree.rightbranch,indent+'  ')
        print (indent+'Left->')
        printtree(tree.leftbranch,indent+'  ')

def movielist(rows):
    x=rows[:,0]
    return x

def decisiontree(rows):

    curent=entropy(rows)
    gain=0
    attr=0
    split_val=0
    rbranch=None
    lbranch=None
    col=rows.shape[1]-1
    for i in range(1,col):
        col_values=rows[:,i]
        unique=np.unique(col_values)
        for val in unique:

            sr,sl=split(rows,i,val)
            p=sr.shape[0]/float(rows.shape[0])
            g=curent - p*entropy(sr) - (1-p)*entropy(sl)
            if g>gain and sr.shape[0]>0 and sl.shape[0]>0:
                gain=g
                attr=i
                split_val= val
                rbranch=sr
                lbranch=sl

    if gain>0:
        rightbranch=decisiontree(rbranch)
        leftbranch=decisiontree(lbranch)
        return Node(cols=attr,vals=split_val,rbr=rightbranch,lbr=leftbranch)
    else:
        return Node(mov=movielist(rows))

def predict(row,tree):
    if(len(tree.images)!=0):
        return tree.images
    else:
        if(row[0][tree.column]>=tree.value):
            return predict(row,tree.rightbranch)
        else:
            return predict(row,tree.leftbranch)
