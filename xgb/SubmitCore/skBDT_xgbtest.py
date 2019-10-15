import matplotlib
matplotlib.use('pdf')
import os
import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
# this wrapper makes it possible to train on subset of features
from rep.estimators import SklearnClassifier
import root_numpy
from optparse import OptionParser
from rep.report.metrics import RocAuc
from sklearn.externals import joblib
from sklearn.utils.class_weight import compute_sample_weight
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc
from plotting_variables import plot_roc
from plotting_variables import plot_classifier_test
from plotting_variables import plot_purity_efficiency
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from plotting_variables import plot_prec_recall_vs_tresh
from plotting_variables import compare_purity_efficiency
from plotting_variables import simple_plot
from math import sqrt


def plot_distribution(data_frame, var_name1='Bmass', var_name2='l2pt', bins=40):
    """The function to plot 2D distribution histograms"""
    plt.hist2d(data_frame[var_name1], data_frame[var_name2], bins = 40, cmap=plt.cm.Blues)
    plt.xlabel(var_name1)
    plt.ylabel(var_name2)
    plt.colorbar()

def compare_train_test(clf,Xtrain,Ytrain,Xtest,Ytest,bins=30,name='test',xlabel='Classifier Output',ylabel='arbitrary units',grid=True,useProb=False):
    decisionsTrain=[]; decisionsTest=[];
    if not useProb:
       decisionsTrain=clf.decision_function(Xtrain)
       decisionsTest=clf.decision_function(Xtest)
    else:
       decisionsTrainTemp=clf.predict_proba(Xtrain)
       decisionsTestTemp=clf.predict_proba(Xtest)
       decisionsTrain=[x[1] for x in decisionsTrainTemp]
       decisionsTest=[x[1] for x in decisionsTestTemp]
    decisionsTrainB=[] ; decisionsTrainS=[] 
    decisionsTestB=[] ; decisionsTestS=[]    
    #print decisionsTrain
    for pred,y in zip(decisionsTrain,Ytrain):
       if y==1: decisionsTrainS.append(pred)
       if y==0: decisionsTrainB.append(pred)
    for pred,y in zip(decisionsTest,Ytest):
       if y==1: decisionsTestS.append(pred)
       if y==0: decisionsTestB.append(pred)
    #print decisionsTrainS 
    plot_classifier_test(decisionsTrainS,decisionsTrainB,decisionsTestS,decisionsTestB,bins=bins,name=name,xlabel=xlabel,ylabel=ylabel,grid=grid)
   
def purity_efficiency(clf,Xtest,Ytest,weights=None): 
    predictions=clf.predict(Xtest)
    if weights is None:
       return precision_score(Ytest,predictions)*recall_score(Ytest,predictions)
    else:
       return precision_score(Ytest,predictions,sample_weight=weights)*recall_score(Ytest,predictions,sample_weight=weights)


def calculate_significance_efficiency_signeff(truthLabels,decisions,thresholds,sgnWeight=1,bkgWeight=1,name="test"):
   Significance=[]; Efficiency=[]; SignEff=[]
   newvalue=decisions; newlabel=truthLabels
   denEff=list(truthLabels).count(1)
   #reduce size
   maxval = max(thresholds)
   newthresholds = [val for ind,val in enumerate(thresholds) if ind%100==0 and val<maxval]
   for thr in newthresholds:
     newvalue,newlabel = zip(*((dec,lab) for dec,lab in zip(newvalue,newlabel)  if dec>thr))
     sgns=float(newlabel.count(1)); bkgs=float(newlabel.count(0))
     Significance.append(sgnWeight*sgns/sqrt(sgns*sgnWeight+bkgs*bkgWeight))
     Efficiency.append(sgns/denEff)
     SignEff.append(Significance[-1]*Efficiency[-1])
   #results
   print "max significance ",max(Significance)," threshold ",newthresholds[Significance.index(max(Significance))]
   print "max eff ",max(Efficiency)," threshold ",newthresholds[Efficiency.index(max(Efficiency))]
   print "max significance*efficiency ",max(SignEff)," threshold ",newthresholds[SignEff.index(max(SignEff))]

   #plots
   simple_plot(newthresholds,Significance,varX="Classifier Output",varY="Significance",name="significance_"+name)
   simple_plot(newthresholds,Efficiency,varX="Classifier Output",varY="Efficiency",name="efficiency_"+name)
   simple_plot(newthresholds,SignEff,varX="Classifier Output",varY="Significance*Efficiency",name="signEff_"+name)





   
if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("--model", dest="model", default='adaboost', type="str", help="full model name")
  parser.add_option("--nevts",dest="nEvents", default=1.0, type="float",help=" ")
  (options, args) = parser.parse_args()
  print options.model+".pkl"
  modelname=options.model
  options.model=options.model+".pkl"
  if not os.path.exists(options.model):
     print "model",options.model," not exist"
     exit() 
  #read var
  used_columns = """Bprob,BLxy,BeLxy,l1pt,l2pt,kpt,Bcos,Bpt,Beta""".split(",")
  used_columns = [c.strip() for c in used_columns]
  # data load test_sig.root test_bkg.root
  trainsignal=root_numpy.root2array('train_sig.root', treename='mytree',branches=used_columns)
  trainbackgr= root_numpy.root2array('train_bkg.root', treename='mytree',branches=used_columns)
  testsignal= root_numpy.root2array('test_sig.root', treename='mytree',branches=used_columns)
  testbackgr= root_numpy.root2array('test_bkg.root', treename='mytree',branches=used_columns)
  trainsignal=root_numpy.rec2array(trainsignal)
  trainbackgr=root_numpy.rec2array(trainbackgr)
  Xtrain=np.concatenate((trainsignal,trainbackgr))
  Ytrain=np.concatenate(([1 for i in range(len(trainsignal))],[0 for i in range(len(trainbackgr))]))
  testsignal=root_numpy.rec2array(testsignal)
  testbackgr=root_numpy.rec2array(testbackgr)
  Xtest=np.concatenate((testsignal,testbackgr))
  Ytest=np.concatenate(([1 for i in range(len(testsignal))],[0 for i in range(len(testbackgr))]))
  clf=joblib.load(options.model)
  compare_train_test(clf,Xtrain,Ytrain,Xtest,Ytest,bins=30,name=modelname,xlabel='Classifier Output',ylabel='arbitrary units',grid=True,useProb=True)
  weightSgn=0;
  for wg in Ytest:
    if wg==1: weightSgn+=1.0
  weightSgn=options.nEvents*0.4*0.8*4.4*(10**(-7))*0.14/weightSgn # for mu
 # weightSgn=options.nEvents*0.4*0.8*4.4*(10**(-7))*0.088/weightSgn # for e
  clf=joblib.load(options.model)
  
  decisionsTemp=clf.predict_proba(Xtest)
  decisions=[x[1] for x in decisionsTemp]

  weightTest=[1 if w==0 else weightSgn for w in Ytest]  
  print "Results for model ",options.model
  precision, recall, thresholds = precision_recall_curve(Ytest,decisions,sample_weight=weightTest)
  plot_purity_efficiency(recall,precision*recall,name=modelname)
  plot_prec_recall_vs_tresh(precision,recall,thresholds,name=modelname)
  calculate_significance_efficiency_signeff(Ytest,decisions,thresholds,weightSgn,name=modelname)
  fpr,tpr,thresholds=roc_curve(Ytest,decisions)
  roc_auc= auc(fpr,tpr)
  plot_roc(fpr,tpr,roc_auc,name=modelname)

