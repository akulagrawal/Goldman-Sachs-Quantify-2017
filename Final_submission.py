
# coding: utf-8

# In[189]:

import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt


# In[190]:

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search


# In[191]:

train=pd.DataFrame(pd.read_csv('gcTrianingSet.csv'))
train2=pd.DataFrame(pd.read_csv('gcTrianingSet.csv'))
prediction=pd.DataFrame(pd.read_csv('gcPredictionFile.csv'))


# In[192]:

train.head


# In[193]:

train.keys()


# In[194]:

train.shape


# In[195]:

train.describe()


# In[196]:

np.sum(train['gcInitialMemory']==0)


# In[197]:

features=list(train.keys())


# In[198]:

featuresPlot=list(train.keys())


# In[199]:

featuresPlot.remove('query token')


# In[200]:

features


# In[201]:

featuresPlot


# In[202]:

train['query token']


# In[46]:

for i in featuresPlot:
    plt.hist(train[i])
    plt.xlabel(i)
    plt.show()


# In[485]:

plt.hist(np.cbrt(mergeGc['cpuTimeTaken']))
plt.show()


# In[67]:

plt.hist(np.power(train['userTime'],1))
plt.show()


# In[203]:

plt.hist( (train['userTime']-np.mean(train['userTime']))/np.std(train['userTime']) )
plt.show()


# In[204]:

np.mean((train['userTime']-np.mean(train['userTime']))/np.std(train['userTime']))


# In[205]:

np.max(train['finalFreeMemory']+train['finalUsedMemory'])


# In[86]:

train['initialFreeMemory']+train['initialUsedMemory']


# In[87]:

train['gcFinalMemory']-train['gcInitialMemory']+train['gcFinalMemory']


# In[ ]:

np.sum[]


# In[89]:

train['gcTotalMemory'].unique()


# In[90]:

train['gcInitialMemory']


# In[206]:

train['query token'].unique()


# In[207]:

train2=train


# In[208]:

train2.insert(0,'queryno',train['query token'].str.split('_').str[1])


# In[209]:

train.keys()==train2.keys()


# In[210]:

train2.keys()


# In[212]:

train2.keys()


# In[213]:

queryToken =train2.pop('query token')


# In[214]:

train2.keys()


# In[215]:

gc=train2[train2['gcRun']==True]


# In[216]:

np.sum(gc['gcInitialMemory']==0)


# In[217]:

len(featuresPlot)


# In[218]:

len(gc.keys())


# In[219]:

featuresPlot2=featuresPlot


# In[220]:

featuresPlot2.append("queryno")


# In[380]:

pearsonr(gc['gcInitialMemory'],gc['initialUsedMemory'])


# In[363]:

for i in featuresPlot2:
    plt.scatter(gc['gcInitialMemory'],gc[i])
    plt.xlabel('gcIni')
    plt.ylabel(i)
    plt.show()


# In[364]:

for i in featuresPlot2:
    plt.scatter(gc2['gcInitialMemory'],gc2[i])
    plt.xlabel('gcIni')
    plt.ylabel(i)
    plt.show()


# In[222]:

featuresPlot2


# In[359]:

from scipy.stats import pearsonr


# In[360]:

pearsonr(gc['gcFinalMemory'],gc['finalUsedMemory'])


# In[223]:

for i in featuresPlot2:
    plt.scatter(gc['gcFinalMemory'],gc[i])
    plt.xlabel('gcFin')
    plt.ylabel(i)
    plt.show()


# In[ ]:

plt.scatter(gc['gcFinalMemory'], )


# In[224]:

for i in featuresPlot2:
    plt.scatter(gc['gcTotalMemory'],gc[i])
    plt.xlabel('gcTotal')
    plt.ylabel(i)
    plt.show()


# In[225]:

gc=gc[gc['gcTotalMemory']>6.5]


# In[226]:

#after outlier
for i in featuresPlot2:
    plt.scatter(gc['gcTotalMemory'],gc[i])
    plt.xlabel('gcTotal')
    plt.ylabel(i)
    plt.show()


# In[238]:

plt.scatter(gc['gcTotalMemory'], np.multiply(gc['gcInitialMemory'],1/gc['gcFinalMemory']) )
plt.show()


# In[ ]:

plt.scatter(gc['gcTotalMemory'], np.power(gc['gcInitialMemory']+gc['gcFinalMemory']) )
plt.show()


# In[252]:

train2['initialFreeMemory'][16]+train2['initialUsedMemory'][16]-train2['gcTotalMemory'][14]


# In[242]:

train2['gcRun']


# In[381]:

plt.scatter(gc['gcTotalMemory'], (gc['initialUsedMemory']+gc['initialFreeMemory']) )
plt.show()


# In[439]:

pearsonr(gc['gcTotalMemory'], (gc['initialUsedMemory']+gc['initialFreeMemory']) )


# In[259]:

gc=gc[gc['initialFreeMemory']+gc['initialUsedMemory']>6.5]


# In[382]:

#outlier removed
plt.scatter(gc['gcTotalMemory'], (gc['finalUsedMemory']+gc['finalFreeMemory']) )
plt.show()


# In[271]:

np.mean(gc['gcTotalMemory']-(gc['initialUsedMemory']+gc['initialFreeMemory']))


# In[278]:

plt.scatter((gc['finalUsedMemory']+gc['finalFreeMemory']), (gc['initialUsedMemory']+gc['initialFreeMemory']) )
plt.show()


# In[383]:

plt.scatter((train2['finalUsedMemory']+train2['finalFreeMemory']), (train2['initialUsedMemory']+train2['initialFreeMemory']) )
plt.show()


# In[281]:

gc.to_csv('gc.csv')


# In[288]:

train2.to_csv('train2.csv')


# In[289]:

gc2=pd.DataFrame(pd.read_csv('train2.csv'))


# In[290]:

gc2=gc2[gc2['gcRun']==False]


# In[291]:

plt.scatter((gc2['finalUsedMemory']+gc2['finalFreeMemory']), (gc2['initialUsedMemory']+gc2['initialFreeMemory']) )
plt.show()


# In[261]:

gc['initialFreeMemory']-gc['gcInitialMemory']


# In[274]:

(gc['finalFreeMemory']+gc['finalUsedMemory']).unique()


# In[276]:

np.mean(train2['initialFreeMemory']+train2['initialUsedMemory']-(train2['finalFreeMemory']+train2['finalUsedMemory'] ) )


# In[ ]:

np.mean(train2['initialFreeMemory']+train2['initialUsedMemory']-(train2['finalFreeMemory']+train2['finalUsedMemory'] ) )


# In[294]:

gc2.keys()


# In[300]:

gc2[gc2['queryno']==1]['cpuTimeTaken']


# In[301]:

gc2[gc2['queryno']==2]['cpuTimeTaken']


# In[302]:

gc2[gc2['queryno']==3]['cpuTimeTaken']


# In[309]:

gc2.groupby('queryno').var()


# In[310]:

gc2.groupby('queryno').mean()


# In[311]:

gc2[gc2['queryno']==23]['cpuTimeTaken']


# In[ ]:

gc[gc['queryno']==3]['cpuTimeTaken']


# In[347]:

np.array(gc.groupby('queryno')['gcInitialMemory'].mean()


# In[348]:

gc2.groupby('queryno')


# In[357]:

gcf=gc2[gc2['queryno']==2]


# In[358]:

np.std(gcf['gcInitialMemory']-gcf['initialFreeMemory'])


# In[ ]:




# In[365]:

gc.shape


# In[366]:

gc2.shape


# In[367]:

gc.shape+gc2.shape


# In[368]:

gc.keys()


# In[369]:

gc2.keys()


# In[371]:

del gc2['Unnamed: 0']


# In[372]:

gc2.keys()


# In[379]:

gc.shape[0]+gc2.shape[0]-train2.shape[0]


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[394]:

#check if correct
plt.scatter((train2['finalUsedMemory']+train2['finalFreeMemory']), (train2['initialUsedMemory']+train2['initialFreeMemory']) )
plt.show()


# In[402]:

#check if correct
plt.scatter((mergeGc['finalUsedMemory']+mergeGc['finalFreeMemory']), (mergeGc['initialUsedMemory']+mergeGc['initialFreeMemory']) )
plt.show()


# In[396]:

mergeGc.to_csv('final1.csv')


# In[407]:

np.min((mergeGc['finalUsedMemory']+mergeGc['finalFreeMemory']))


# In[ ]:




# In[480]:

#check if correct
plt.scatter((mergeGc['finalUsedMemory']+mergeGc['finalFreeMemory']), (mergeGc['initialUsedMemory']+mergeGc['initialFreeMemory']) )
plt.show()


# In[458]:

mm=pd.DataFrame(pd.read_csv('final1.csv'))


# In[459]:

mergeGc=mm


# In[474]:

mergeGc=mergeGc[ (mergeGc['finalUsedMemory']+mergeGc['finalFreeMemory']) >6.5]


# In[481]:

mm.shape


# In[482]:

mergeGc.shape


# In[463]:

mergeGc=mm


# In[464]:

mergeGc.shape


# In[385]:

frames=[gc,gc2]
mergeGc = pd.concat(frames)


# In[386]:

mergeGc.shape


# In[387]:

mergeGc.describe()


# In[483]:

plt.scatter(mergeGc['gcInitialMemory'],mergeGc['initialUsedMemory'])
plt.show()


# In[484]:

mergeGc.to_csv('clean.csv')


# In[486]:

mergeGc['cpuTimeTaken'] = np.cbrt(mergeGc['cpuTimeTaken'])


# In[487]:

plt.hist(mergeGc['cpuTimeTaken'])
plt.show()


# In[ ]:




# In[488]:

mergeGc.to_csv('trans.csv')


# In[496]:

len(features)


# In[495]:

len(mergeGc.keys())


# In[493]:

del mergeGc['Unnamed: 0']


# In[498]:

featuresPlot2


# In[503]:

from sklearn import preprocessing as prep


# In[502]:

mergeGc.keys()


# In[505]:

scaler1 = prep.StandardScaler().fit(mergeGc.initialUsedMemory)
scaler2 = prep.StandardScaler().fit(mergeGc.initialFreeMemory)
scaler4 = prep.StandardScaler().fit(mergeGc.gcInitialMemory)
scaler5 = prep.StandardScaler().fit(mergeGc.gcFinalMemory)
scaler6 = prep.StandardScaler().fit(mergeGc.gcTotalMemory)
scaler7 = prep.StandardScaler().fit(mergeGc.userTime)
scaler8 = prep.StandardScaler().fit(mergeGc.sysTime)
scaler9 = prep.StandardScaler().fit(mergeGc.realTime)
scaler10 = prep.StandardScaler().fit(mergeGc.cpuTimeTaken)
scaler11 = prep.StandardScaler().fit(mergeGc.finalUsedMemory)
scaler12 = prep.StandardScaler().fit(mergeGc.finalFreeMemory)


# In[ ]:




# In[506]:

mergeGc.to_csv('normal.csv')


# In[507]:

mergeGc


# In[ ]:




# In[521]:

#transform leader
leader=pd.DataFrame(pd.read_csv('gcPredictionFile.csv'))


# In[522]:

leader['cpuTimeTaken'] = np.log(leader['cpuTimeTaken'])


# In[523]:

leader.keys()


# In[524]:

leader.insert(0,'queryno',leader['query token'].str.split('_').str[1])


# In[525]:

leader.keys()


# In[526]:

del leader['query token']


# In[527]:

leader.keys()


# In[528]:

leader.to_csv('leader.csv')


# In[ ]:




# In[ ]:




# In[ ]:




# In[536]:

#resources
gc2.iloc[:,12]


# In[537]:

gc2.gcRun


# In[538]:

gc2.insert(13,'resources', gc2.finalUsedMemory-gc2.initialUsedMemory )


# In[546]:

gc2.groupby('queryno').mean()


# In[577]:

res2 = dict(gc2.groupby('queryno')['resources'].mean())


# In[580]:

res2[8]


# In[581]:

res[8]


# In[564]:

res.keys()


# In[566]:

mean=0


# In[567]:

for i in res.keys():
    mean=mean+res[i]


# In[568]:

mean


# In[569]:

mean=mean/91


# In[570]:

var=0


# In[571]:

for i in res.keys():
    var = var+ (res[i]-mean)**2


# In[572]:

var


# In[573]:

std = var**0.5


# In[574]:

std


# In[575]:

for i in res.keys():
    res[i] = (res[i]-mean)/std


# In[576]:

res


# In[551]:

mergeGc.keys()


# In[552]:

leader.keys()


# In[550]:

mergeGc[['queryno','initialUsedMemory', 'initialFreeMemory', 'cpuTimeTaken', 'gcRun']]


# In[ ]:

#res is normalised
#res2 is not normalised


# In[595]:

trainX = mergeGc[['queryno','initialUsedMemory', 'initialFreeMemory', 'cpuTimeTaken', 'gcRun']]


# In[596]:

trainX.keys()


# In[597]:

trainX.insert(4, 'resources', 0)


# In[598]:

trainX


# In[602]:

for i in np.arange(trainX.shape[0]):
    trainX['resources'].iloc[i] = res[trainX['queryno'].iloc[i]]


# In[605]:

res[10]


# In[603]:

trainX


# In[639]:

trainX.shape


# In[640]:

train.columns


# In[675]:

del train['queryno']


# In[676]:

train.columns


# In[677]:

#xbg begins!!!


# In[678]:

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search


# In[679]:

import time


# In[680]:

import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# In[681]:

train=trainX
target='gcRun'


# In[682]:

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        #xgb_param['num_class']=2
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
             early_stopping_rounds=early_stopping_rounds,stratified=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    #print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
                    
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')



# In[683]:

#Choose all predictors except target & IDcols
t=time.time()
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)

print('time',time.time()-t)


# In[684]:

np.sum([train['gcRun']==True])


# In[685]:

169/2552


# In[ ]:




# In[ ]:




# In[686]:

leader.columns


# In[687]:

leader.insert(4, 'resources', 0)


# In[688]:

leader


# In[703]:

for i in np.arange(leader.shape[0]):
    leader['resources'].iloc[i] = res[int(leader['queryno'].iloc[i])]


# In[705]:

leader.shape


# In[706]:

leader


# In[707]:

res[76]


# In[708]:

query = leader.pop['queryno']


# In[709]:

leader.to_csv('leader2.csv')


# In[710]:

leader.keys()==train.keys()


# In[712]:

del leader['gcRun']


# In[725]:

xgb1.predict(leader).shape


# In[717]:

out = pd.DataFrame(pd.read_csv('leader2.csv'))


# In[718]:

out.columns


# In[719]:

out['gcRun']=xgb1.predict(leader)


# In[720]:

out.shape


# In[721]:

out.to_csv('out1.csv')


# In[722]:

queryLeader = pd.DataFrame(pd.read_csv('gcPredictionFile.csv'))


# In[723]:

queryLeader


# In[724]:

leader.shape


# In[727]:

out


# In[ ]:




# In[ ]:




# In[ ]:




# In[734]:

test  = pd.DataFrame(pd.read_csv('gcPredictionFile.csv'))


# In[728]:

#free begins
free  = pd.DataFrame(pd.read_csv('out1.csv'))


# In[729]:

free


# In[730]:

free.insert(5,'gcInitialMemory', free['initialUsedMemory'])


# In[735]:

free.shape


# In[736]:

test.shape


# In[739]:

test.insert(0,'queryno',test['query token'].str.split('_').str[1])


# In[740]:

for i in np.arange(free.shape[0]):
    free['resources'].iloc[i] = res2[int(test['queryno'].iloc[i])]


# In[741]:

free


# In[742]:

res2[53]


# In[ ]:

free['resources']=res2


# In[744]:

free.columns


# In[745]:

free.insert(6, 'gcTotalMemory', free['initialFreeMemory']+free['initialUsedMemory'])


# In[751]:

free.insert(7, 'finalUsedMemory', free['resources']+free['initialUsedMemory'])


# In[752]:

free


# In[753]:

free.insert(8, 'finalFreeMemory',  free['initialFreeMemory']+free['initialUsedMemory']- free['finalUsedMemory'])


# In[754]:

free


# In[755]:

free.to_csv('out3.csv')


# In[817]:

free=pd.DataFrame(pd.read_csv('out3.csv'))


# In[818]:

free.columns


# In[819]:

free


# In[765]:

free.shape[0]


# In[766]:

np.arange(1,free.shape[0])


# In[767]:

for i in np.arange(1,free.shape[0]):
    free['initialUsedMemory'].iloc[i]=free['finalUsedMemory'].iloc[i-1]
    free['initialFreeMemory'].iloc[i]=free['finalFreeMemory'].iloc[i-1]
    free['gcInitialMemory'].iloc[i] = free['initialUsedMemory'].iloc[i]
    free['gcTotalMemory'].iloc[i] = free['initialFreeMemory'].iloc[i]+free['initialUsedMemory'].iloc[i]
    free['finalUsedMemory'].iloc[i] = free['resources'].iloc[i]+free['initialUsedMemory'].iloc[i]
    free['finalFreeMemory'].iloc[i] = free['initialFreeMemory'].iloc[i]+free['initialUsedMemory'].iloc[i]- free['finalUsedMemory'].iloc[i]


# In[770]:

free.to_csv('out4.csv')


# In[771]:

free.to_csv('out5.csv')


# In[772]:

free


# In[784]:

mergeGc['initialUsedMemory']


# In[786]:

train.shape


# In[792]:

mergeGc['initialUsedMemory'].reshape(1,-1)


# In[805]:

#logreg
from sklearn.linear_model import LinearRegression
#for c in np.arange(1,10,1):
#    log = LogisticRegression(C=c).fit(mergeGc['initialUsedMemory'],mergeGc['gcInitialMemory'])
#    print(log.score(xtest,ytest))
linGcIni = LinearRegression().fit(mergeGc['initialUsedMemory'].values.reshape(-1,1),mergeGc['gcInitialMemory'].values.reshape(-1,1))
#print('final:',log.score((xtest,ytest))
#print('final train:',log.score(xtrain,ytrain))


# In[808]:

trainX.shape


# In[809]:

mergeGc.shape


# In[811]:

linFinalUsed = LinearRegression().fit(trainX['resources'].reshape(-1,1)+mergeGc['initialUsedMemory'].reshape(-1,1), mergeGc['finalUsedMemory'].reshape(-1,1))


# In[813]:

linGcTotal = LinearRegression().fit(mergeGc['initialUsedMemory'].reshape(-1,1)+mergeGc['initialFreeMemory'].reshape(-1,1), mergeGc['gcTotalMemory'] )


# In[816]:

linFinalFree = LinearRegression().fit(mergeGc['initialFreeMemory'].reshape(-1,1)+mergeGc['initialUsedMemory'].reshape(-1,1)-mergeGc['finalUsedMemory'].reshape(-1,1), mergeGc['finalFreeMemory'].reshape(-1,1) )


# In[824]:

for i in np.arange(1,free.shape[0]):
    free['initialUsedMemory'].iloc[i]=free['finalUsedMemory'].iloc[i-1]
    free['initialFreeMemory'].iloc[i]=free['finalFreeMemory'].iloc[i-1]
    free['gcInitialMemory'].iloc[i] = linGcIni.predict(free['initialUsedMemory'].iloc[i])
    free['gcTotalMemory'].iloc[i] = linGcTotal.predict(free['initialFreeMemory'].iloc[i]+free['initialUsedMemory'].iloc[i])
    free['finalUsedMemory'].iloc[i] = linFinalUsed.predict(free['resources'].iloc[i]+free['initialUsedMemory'].iloc[i])
    free['finalFreeMemory'].iloc[i] = linFinalFree.predict(free['initialFreeMemory'].iloc[i]+free['initialUsedMemory'].iloc[i]- free['finalUsedMemory'].iloc[i]) 


# In[826]:

free.to_csv('output6.csv')


# In[827]:

free.columns


# In[831]:

finalOutput=pd.DataFrame(pd.read_csv('output6.csv'))


# In[833]:

finalOutput.shape[0]


# In[834]:

free.shape


# In[837]:

finalOutput.insert(5, 'serialNum', np.arange(1,finalOutput.shape[0]+1))


# In[841]:

finalOutput['initialFreeMemory']= free['finalFreeMemory']


# In[843]:

finalOutput['gcRun']=xgb1.predict(leader)


# In[844]:

finalOutput.columns


# In[845]:

del finalOutput['Unnamed: 0']
del finalOutput['initialUsedMemory']
del finalOutput['cpuTimeTaken']
del finalOutput['resources']
del finalOutput['gcInitialMemory']
del finalOutput['gcTotalMemory']
del finalOutput['finalFreeMemory']
del finalOutput['finalUsedMemory']


# In[847]:

columnTitles=['serialNum','initialFreeMemory', 'gcRun']


# In[848]:

finalOutput= finalOutput.reindex(columns=columnTitles)


# In[854]:

finalOutput


# In[856]:

finalOutput.to_csv('output.csv', index=False)


# In[ ]:



