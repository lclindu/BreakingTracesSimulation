
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg') ###???? does it work;???
import matplotlib.pyplot as plt
from scipy import stats
import time

import sklearn.metrics.pairwise as skmp
import matplotlib.cm as cm

np.random.seed(429)

t_start1 = time.clock()

# ***************parameter-initialization*********************
runs = 4000

data_points = 500
d_lower_limit = 0.0
d_upper_limit = 4
d = np.linspace(d_lower_limit, d_upper_limit, data_points)


G0 = 1;  #77.84
betaT = 10.00#

mean_snap_back_GG = 0.5
mean_snap_back_GM = 0.1 

#standard deviation
std = 0.1
# junction formation probability
prj = 0.8 ###random [0.5, 0.9]

# gdMatrix = pd.DataFrame({'distance': [],
#                               'logG': [] })


label_junc_or_not = np.ndarray(runs,dtype=int)
label_junc = np.ndarray(runs, dtype=int)

Curves_lenth = np.ndarray(runs, dtype=int)  ##for snap;


pm = runs
qn, = np.shape(d)

dataCondu = np.zeros((pm, qn),dtype=float)  ###///may not used??
# dataDist = np.zeros((pm, qn),dtype=float)   ####////
dd = runs * d.tolist()
dataDist = np.array(dd).reshape(pm, qn)

cbins = 100;

histrange = (-8, 0)

del dd


ngroup = 1000
highClasses = 10
lowClasses = 2


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[28]:


def g_mt_function(x, A, B, lm, n, junc_or_not):
    Gt = G0 * np.exp(-betaT * x)
    Gm = A * G0 * np.cos(np.arcsin(x/lm))** (n) + G0 * B # if x/lm < 1 else 1
    gmt = Gt + Gm * junc_or_not  # np.arcsin(x/lm)
    return gmt

# Gt + Gtr
def g_ts_function(x, A, B, lm, junc_or_not):
    Gt = G0 * np.exp(-betaT * x)
    Gtr = G0 * np.exp(-betaT * (x- lm * 0.6))  # * 0.6 change to + 0.6
    GM = G0 * B
    gts = Gt + GM * Gtr * junc_or_not
    return gts 

def g_sensitive():
    g_sens = np.random.normal(-8.2, 0.5) #####need random?
    return g_sens


# remove data in 2 snap-backs region
def remove_snap_or_not(x, y, z = False):
    pd.options.mode.chained_assignment = None
    singleMatrix = pd.DataFrame({'distance': x,
                                'logG': y})
    if z:
        snap_back_GG = gaussian_dis(mean_snap_back_GG)
        snap_back_GM = gaussian_dis(mean_snap_back_GM)    

        # remove data points in Au-Au snap-back
        singleMatrix1 = singleMatrix[(singleMatrix['distance'] >                                       snap_back_GG) & (singleMatrix['distance'] < snap_back_GG + lm)]
        # z + Au-Au snapback
        singleMatrix1['distance'] -= snap_back_GG # mean_
        # remove data points in Au-molecule snapback
        singleMatrix2 = singleMatrix[singleMatrix['distance'] > snap_back_GG + lm + snap_back_GM]
        # z + Au-molecule snapback
        singleMatrix2['distance'] -= snap_back_GG + mean_snap_back_GM  # mean_
        return pd.concat([singleMatrix1, singleMatrix2])
    else:
        return singleMatrix

# def gaussian_dis(para):
#     return np.random.normal(para, para * 0.1)


def noise_g(x):
    noise = np.random.normal(0, x * 0.06)  #####how to set resonable??
    while abs(noise) > x/2:
        noise = np.random.normal(0, x * 0.06)
    return noise


def ConstructCurves(Aarray_C, Barray_C,narray_C,mu_lmarray_C):
    label_junc_or_not_out = np.ndarray(runs, dtype=np.int)
    dataCondu_out = np.zeros((pm, qn),dtype=float)
    g_single = np.ndarray(qn)
    prj = np.random.uniform(0.5, 0.9)
    for i in range(runs):
        junc_or_not = ((np.random.random() < prj) and 1 or 0)
        index_i  = np.random.randint(runs * runs) % len(Aarray_C)
#         print(i, index_i, len(Aarray))
        A = Aarray_C[index_i]
        B = Barray_C[index_i]
        n = narray_C[index_i]
        mu_lm = mu_lmarray_C[index_i]

        lm =  np.random.normal(mu_lm, mu_lm * 0.05)#gaussian_dis(mu_lm)#mu_lm
#         rnd_factor = np.random.random()

        for (j, qni) in zip(d, range(qn)):
            if 0 <= j <= lm: #分子结未断裂
                g_scalar1 = g_mt_function(j, A, B, lm, n, junc_or_not)
                log_scalar1 = np.log10(g_scalar1 / G0)
                g_scalar = log_scalar1 + noise_g(abs(log_scalar1))

            elif lm < j <= d_upper_limit: #分子结已断裂
                g_scalar2 = g_ts_function(j, A, B, lm, junc_or_not)
                log_scalar2 = np.log10(g_scalar2 / G0)
                g_scalar = log_scalar2 + noise_g(abs(log_scalar2))
            if g_scalar < -8.10:
                g_scalar = g_sensitive()

            g_single[qni] = g_scalar

        logG_single = g_single#np.log10(g_single / G0)
#         traceMatrix = remove_snap_or_not(d, logG_single, False) #已移除snapBack的单条曲线

        dataCondu_out[i,:] = logG_single
        label_junc_or_not_out[i] = index_i+1 if junc_or_not else junc_or_not


    # *********** -- end --************************************
    t_end1 = time.clock() - t_start1
    # x_D and y_logG are ready for plot!
    print('******************************')
    print("* 计算完毕，共耗时 " + str(round(t_end1, 2)) + " 秒！*")
    print('******************************')
    return dataCondu_out, label_junc_or_not_out

def Compute2dPairwiseDistance(dataConduOut, labelsofClusters):   ###only use once??
    labelsmax = len(set(labelsofClusters))  #  np.int(label_junc_or_not.max()) + 1
    j = 0
    
    Hist2dXY = np.ndarray([labelsmax, cbins*cbins])  ##bins = 100;;;

    for i in range(labelsmax):
        index_i = np.array(np.where(labelsofClusters == i)).reshape(-1)
        if index_i.shape[0] < 3:
            continue

        ClusterI_x = dataDist[index_i]
        ClusterI = dataConduOut[index_i]
        ClusterI_2dHist = np.histogram2d(ClusterI_x.reshape(-1), ClusterI.reshape(-1), cbins, density=True)
        Hist2dXY[j, :] = ClusterI_2dHist[0].reshape(-1,)
        j += 1 
    return skmp.cosine_distances(Hist2dXY[:labelsmax, :])

# DistThreshHold = np.array(np.where(cosDist < 0.05))
# print(DistThreshHold)
# for i in range(len(DistThreshHold.T)):
#     label1 = DistThreshHold[0, i]
#     label2 = DistThreshHold[1, i]
#     if label1 < label2:
#         Label_junc[np.where(Label_junc == label1)] = label2

def plot2dCloudofClusters(Label_junc, DataCondu_out):    
    labelsmax = len(set(Label_junc)) 
    j = 0
    print(labelsmax)
    for i in set(Label_junc):
        plt.figure(labelsmax, figsize=(3*labelsmax, 5))

        index_i = np.array(np.where(Label_junc == i)).reshape(-1)
    #     print(i)
        if index_i.shape[0] < 3:
            continue

        ClusterI_x = dataDist[index_i]
        ClusterI = DataCondu_out[index_i]

        plt.subplot(1, labelsmax, j+1)
        j += 1
        plt.title('OriData %s %s' %(i, index_i.shape[0]))
        ClusterI_2dHist = plt.hist2d(ClusterI_x.reshape(-1), 
                                       ClusterI.reshape(-1), cbins, 
                             cmap=cm.coolwarm, normed = True,vmax=0.6, vmin=0.00)
    return labelsmax

def plot2dCloudandSave(DataCondu_out, pathname):

    mD, nD  = DataCondu_out.shape
    plt.figure(figsize=(2 * 6, 5))
    plt.subplot(1,2, 1)
    Condu_2dHist = plt.hist2d(dataDist[:mD].reshape(-1), DataCondu_out.reshape(-1), cbins, 
                             cmap=cm.coolwarm, normed = True,vmax=0.7, vmin=0.01)
    plt.title('All data 2d Cloud')
    plt.subplot(1,2,2)
    histDist = np.histogram(DataCondu_out, cbins, range=histrange)
    plt.plot(histDist[1][:-1], histDist[0])
    plt.title('All data histogram')
    plt.savefig(pathname, dpi=600)
    plt.close()
    


# In[7]:



np.random.randint(runs * runs)


# In[27]:


# 4/499;;;;+++++


# In[8]:


np.random.seed(429)
Classes = np.random.randint(lowClasses, highClasses+1, ngroup)
np.random.seed(np.random.randint(runs * runs))


# In[20]:


DistConstrain = 0.05
def ConstructGroup(indexrand):
    
    np.random.seed(np.random.randint(1000000000))
        
#     indexintA = np.random.randint(ngroup**3+1, size=Classes[indexrand])
#     indexintB = np.random.randint(ngroup**3+1, size=Classes[indexrand])
#     indexintA.sort()
#     indexintB.sort()
#     Aarrayrand = Aspace[indexintA]
#     Barrayrand = Bspace[indexintB]
    Aarrayrand = np.random.uniform(np.log10(3.8405e-08), np.log10(0.2), Classes[indexrand])
    Barrayrand = np.random.uniform(np.log10(7.8576e-09), np.log10(0.0594), Classes[indexrand])
    Aarrayrand = np.power(10, Aarrayrand)
    Barrayrand = np.power(10, Barrayrand)
    Aarrayrand.sort()
    Barrayrand.sort()
    narrayrand = np.random.uniform(3.50949, 4.5403, Classes[indexrand])
#     indexintmu_lm = np.random.randint(ngroup+1, size=Classes[indexrand])
#     mu_lmarrayrand = mu_lmspace[indexintmu_lm]
    mu_lmarrayrand = np.random.uniform(0.5, 3.5, Classes[indexrand])
    mu_lmarrayrand.sort()
    mu_lmarrayrand = mu_lmarrayrand[::-1]

    DataCondu_out, Label_junc = ConstructCurves(Aarrayrand, Barrayrand,narrayrand,mu_lmarrayrand)
#     print(Label_junc)
    np.savetxt('/mnt/home/%s/SimulationData%s/Ori/dataCondu/Conductance%s_20190117.txt' %(username,DistConstrain,indexrand), DataCondu_out, fmt='%0.5f')
    np.savetxt('/mnt/home/%s/SimulationData%s/Ori/labels/Labels%s_20190117.txt' %(username,DistConstrain,indexrand), Label_junc, fmt='%d')
    
    cosDist  = Compute2dPairwiseDistance(DataCondu_out, Label_junc)
    
    pathname = '/mnt/home/%s/SimulationData%s/Ori/Image/CloundandHist%s_20190117.png' %(username,DistConstrain, indexrand)
    plot2dCloudandSave(DataCondu_out, pathname)
    
    DistThreshHold = np.array(np.where(cosDist < DistConstrain))   ### delete or replace?? choose delete ??
    del_index = np.array([], dtype=np.int)
    for i in range(len(DistThreshHold.T)):
        label1 = DistThreshHold[0, i]
        label2 = DistThreshHold[1, i]
        if label1 < label2:
            del_indextemp = np.where(Label_junc == label2)
            del_index = np.append(del_index, del_indextemp)
    Label_junc = np.delete(Label_junc, del_index)
    
    
    ######make label_junc from 0 to len(set(Label_junc))
#     setlabel = set(Label_junc)
#     labelNum = len(setlabel)
#     a = set(range(labelNum))
#     lostNum = a - setlabel
#     moreNum = setlabel - a
#     for i,j in zip(lostNum, moreNum):
#         Label_junc[np.where(Label_junc == j)] = i
    
    DataCondu_out = np.delete(DataCondu_out, del_index, 0)
#     print(Label_junc)
    np.savetxt('/mnt/home/%s/SimulationData%s/del/dataCondu/ConductanceD%s_20190117.txt' %(username, DistConstrain,indexrand), DataCondu_out,fmt='%0.5f')
    np.savetxt('/mnt/home/%s/SimulationData%s/del/labels/LabelsD%s_20190117.txt' %(username,DistConstrain, indexrand), Label_junc, fmt='%d')
    
    pathname = '/mnt/home/%s/SimulationData%s/del/Image/CloundandHistD%s_20190117.png' %(username, DistConstrain,indexrand)
    plot2dCloudandSave(DataCondu_out, pathname)


# In[ ]:





# In[29]:


import pwd
import os
# username = os.getlogin()
username = pwd.getpwuid(os.getuid()).pw_name
OriDataConduPath = '/mnt/home/%s/SimulationData%s/Ori/dataCondu/'%(username, DistConstrain)
OriDataImagePath = '/mnt/home/%s/SimulationData%s/Ori/Image/'%(username, DistConstrain)
OriDatalabelsPath = '/mnt/home/%s/SimulationData%s/Ori/labels/'%(username, DistConstrain)
delDataConduPath = '/mnt/home/%s/SimulationData%s/del/dataCondu/'%(username, DistConstrain)
delDataImagePath = '/mnt/home/%s/SimulationData%s/del/Image/'%(username, DistConstrain)
delDatalabelsPath = '/mnt/home/%s/SimulationData%s/del/labels/'%(username, DistConstrain)
if not os.path.exists(OriDataConduPath):
    os.makedirs(OriDataConduPath)
    os.makedirs(OriDataImagePath)
    os.makedirs(OriDatalabelsPath)
    os.makedirs(delDataConduPath)
    os.makedirs(delDataImagePath)
    os.makedirs(delDatalabelsPath)


# In[16]:


# np.savetxt(OriDataConduPath+'Clusters.txt', Classes, fmt='%d')
plt.hist(Classes, 9)


# In[30]:


import multiprocessing as mp

pool = mp.Pool(processes=30)
results = pool.map(ConstructGroup, range(30))


# In[21]:


for i in range(5):
    DistConstrain = 0.05 * (i+1)
    OriDataConduPath = '/mnt/home/%s/SimulationData%s/Ori/dataCondu/'%(username, DistConstrain)
    OriDataImagePath = '/mnt/home/%s/SimulationData%s/Ori/Image/'%(username, DistConstrain)
    OriDatalabelsPath = '/mnt/home/%s/SimulationData%s/Ori/labels/'%(username, DistConstrain)
    delDataConduPath = '/mnt/home/%s/SimulationData%s/del/dataCondu/'%(username, DistConstrain)
    delDataImagePath = '/mnt/home/%s/SimulationData%s/del/Image/'%(username, DistConstrain)
    delDatalabelsPath = '/mnt/home/%s/SimulationData%s/del/labels/'%(username, DistConstrain)
    for j in range(2):
        Classes = np.random.randint(lowClasses, highClasses+1, ngroup)
        if not os.path.exists(OriDataConduPath):
            os.makedirs(OriDataConduPath)
            os.makedirs(OriDataImagePath)
            os.makedirs(OriDatalabelsPath)
            os.makedirs(delDataConduPath)
            os.makedirs(delDataImagePath)
            os.makedirs(delDatalabelsPath)
        pool = mp.Pool(processes=30)
        results = pool.map(ConstructGroup, range(ngroup))
        strtime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        srcname = '/mnt/home/%s/SimulationData%s/Ori'%(username, DistConstrain)
        srcname2 = '/mnt/home/%s/SimulationData%s/del'%(username, DistConstrain)
        print(srcname, srcname2)
        try:
            os.rename(srcname, srcname+strtime)
        except Exception as e:
            print (e)
            print ('rename dir fail\r\n')
        else:
            print('rename dir success\r\n')

        try:
            os.rename(srcname2, srcname2+strtime)
        except Exception as e:
            print (e)
            print ('rename dir fail\r\n')
        else:
            print('rename dir success\r\n')

