import os
import pandas
import numpy as np
from rep.estimators import SklearnClassifier
import root_numpy
import argparse
from rep.report.metrics import RocAuc
from sklearn.externals import joblib
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from plotting_variables import *
from math import sqrt
import time

   
def purity_efficiency(clf,Xtest,Ytest,weights=None): 
    predictions=clf.predict(Xtest)
    if weights is None:
       return precision_score(Ytest,predictions)*recall_score(Ytest,predictions)
    else:
       return precision_score(Ytest,predictions,sample_weight=weights)*recall_score(Ytest,predictions,sample_weight=weights)

def calculate_significance(truthLabels,decisions,thresholds,sgnWeight=1,bkgWeight=1,name="test"):
   Significance=[];
   for thr in thresholds:
     sgnTemp=0; bkgTemp=0; 
     for value,truth in zip(decisions,truthLabels):
       if value>thr:
          if truth == 1:
            sgnTemp=sgnTemp+sgnWeight
          if truth == 0:
            bkgTemp=bkgTemp+bkgWeight      
     if bkgTemp==0 and sgnTemp==0:
       bkgTemp=1;
     Significance.append(sgnTemp/sqrt(sgnTemp+bkgTemp))
   print "max significance ",max(Significance)," threshold ",thresholds[Significance.index(max(Significance))]
   return Significance
   

def calculate_efficiency(truthLabels,decisions,thresholds,name="test"):
   unique, counts= np.unique(truthLabels,return_counts=True)
   NumEff=[];
   denEff=counts[1]
   for thr in thresholds:
     sgnTemp=0; 
     for value,truth in zip(decisions,truthLabels):
       if value>thr and truth == 1:
         sgnTemp+=1
     NumEff.append(sgnTemp/denEff)
   print "max eff ",max(NumEff)," threshold ",thresholds[NumEff.index(max(NumEff))]
   return NumEff;
   

def calculate_maxsignEff(truthLabels,decisions,thresholds,sgnWeight=1,bkgWeight=1):
   unique, counts= np.unique(truthLabels,return_counts=True)
   SignEff=[];
   denEff=counts[1]
   for thr in thresholds:
     sgnTemp=0; bkgTemp=0; effTemp=0;
     for value,truth in zip(decisions,truthLabels):
       if value>thr or value==thr:
          if truth == 1:
            sgnTemp+=sgnWeight
            effTemp+=1
          if truth == 0:
            bkgTemp+=bkgWeight      
     if bkgTemp==0 and sgnTemp==0:
       print "thr ",thr,max(decisions)
       bkgTemp=1;
     SignEff.append(sgnTemp/sqrt(sgnTemp+bkgTemp)*effTemp/denEff)
   print "max significance*efficiency ",max(SignEff)," threshold ",thresholds[SignEff.index(max(SignEff))]
   return max(SignEff)

def calculate_wght(Ymatrix,expectedSig):
  weightSgn=0;
  for wg in Ymatrix:
    if wg==1: weightSgn+=1.0
  weightSgn=expectedSig/weightSgn
  weightTest=[1 if w==0 else weightSgn for w in Ymatrix]
  return weightSgn

     
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--models", dest="models",nargs='+', default=['adaboost.pkl','gradboost.pkl'], type=str, help="full model name")
  parser.add_argument("--expSigTrain",dest="expectedSigTrain", default=1.0, type=float,help=" ")
  parser.add_argument("--expSigTest",dest="expectedSigTest", default=1.0, type=float,help=" ")
  parser.add_argument("--metric",dest="metric", default="Ntrees", type=str,help=" ")
  parser.add_argument("--inputSigTrain",dest="inputSigTrain", default="train_sig.root", type=str,help=" ")
  parser.add_argument("--inputBkgTrain",dest="inputBkgTrain", default="train_bkg.root", type=str,help=" ")
  parser.add_argument("--inputSigTest",dest="inputSigTest", default="test_sig.root", type=str,help=" ")
  parser.add_argument("--inputBkgTest",dest="inputBkgTest", default="test_bkg.root", type=str,help=" ")
  parser.add_argument("--pathweight",dest="path", default='/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/uboost/CMSSW_10_1_4/src/hep_ml/ScikitWeights_AdaOverT', type=str,help=" ")
  args = parser.parse_args()
  t0 = time.time()
  #read var
  used_columns = """Bprob,BLxy,BeLxy,l1pt,l2pt,kpt,Bcos,Bpt,Beta""".split(",")
  used_columns = [c.strip() for c in used_columns]
  # data load test_sig.root test_bkg.root
  testsignal= root_numpy.root2array(args.inputSigTest, treename='mytree',branches=used_columns)
  testbackgr= root_numpy.root2array(args.inputBkgTest, treename='mytree',branches=used_columns)
  trainsignal=root_numpy.root2array(args.inputSigTrain, treename='mytree',branches=used_columns)
  trainbackgr= root_numpy.root2array(args.inputBkgTest, treename='mytree',branches=used_columns)
  trainsignal=root_numpy.rec2array(trainsignal)
  trainbackgr=root_numpy.rec2array(trainbackgr)
  Xtrain=np.concatenate((trainsignal,trainbackgr))
  Ytrain=np.concatenate(([1 for i in range(len(trainsignal))],[0 for i in range(len(trainbackgr))]))
  testsignal=root_numpy.rec2array(testsignal)
  testbackgr=root_numpy.rec2array(testbackgr)
  Xtest=np.concatenate((testsignal,testbackgr))
  Ytest=np.concatenate(([1 for i in range(len(testsignal))],[0 for i in range(len(testbackgr))]))
  train_weight=calculate_wght(Ytrain,args.expectedSigTrain)
  test_weight=calculate_wght(Ytest,args.expectedSigTest)
  maxMetricTrain=[]; vparams=[]; maxMetricTest=[]
  for model in args.models:
    params=model.split("_")
    for param in params:
       if args.metric in param:          
          vparams.append(float(param[len(args.metric):]))  
    clf=joblib.load(args.path+"/"+model+".pkl")
    prob=True
    if 'XGBoost' in model: 
       prob=True;
    decisionTest=[]; decisionTrain=[]
    if prob:
      decisionTemp=clf.predict_proba(Xtest)
      decisionTest=[x[1] for x in decisionTemp]
      decisionTemp2=clf.predict_proba(Xtrain)
      decisionTrain=[x[1] for x in decisionTemp2]
    else:
      decision=clf.decision_function(Xtest)    
    fpr,tpr,thresholdTest=roc_curve(Ytest,decisionTest)
    fpr,tpr,thresholdTrain=roc_curve(Ytrain,decisionTrain)
    #if "sigEff" in args.metric:
    maxMetricTest.append(calculate_maxsignEff(Ytest,decisionTest,thresholdTest,sgnWeight=test_weight))
    maxMetricTrain.append(calculate_maxsignEff(Ytrain,decisionTrain,thresholdTrain,sgnWeight=train_weight))
     
  legs=["train","test"]; maxes=[maxMetricTrain,maxMetricTest]
  npparams=np.array(vparams)
  print len(npparams),len(maxMetricTrain)
  multi_plot_sameX(npparams,maxes,legs,"plot"+args.metric,args.metric,"Significance*Efficiency")  

