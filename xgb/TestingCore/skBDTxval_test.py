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
    for pred,y in zip(decisionsTrain,Ytrain):
       if y==1: decisionsTrainS.append(pred)
       if y==0: decisionsTrainB.append(pred)
    for pred,y in zip(decisionsTest,Ytest):
       if y==1: decisionsTestS.append(pred)
       if y==0: decisionsTestB.append(pred) 
    plot_classifier_test(decisionsTrainS,decisionsTrainB,decisionsTestS,decisionsTestB,bins=bins,name=name,xlabel=xlabel,ylabel=ylabel,grid=grid)
   
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
   

def calculate_signEff(truthLabels,decisions,thresholds,sgnWeight=1,bkgWeight=1):
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
   return SignEff

def decouple_histo_range(data):
  names=[]; xmins=[]; xmaxs=[]; ymins=[]; ymaxs=[];
  for i in range(0, len(data)):
    if i%5==0: names.append(data[i]); 
    elif i%5==1: xmins.append(float(data[i]))
    elif i%5==2: xmaxs.append(float(data[i]))
    elif i%5==3: ymins.append(float(data[i]))
    elif i%5==4: ymaxs.append(float(data[i]))
  return names,xmins,xmaxs,ymins,ymaxs
     
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--models", dest="models",nargs='+', default=['adaboost.pkl','gradboost.pkl'], type=str, help="full model name")
  parser.add_argument("--exclude", dest="exclude",nargs='+', default=['adaboost','gradboost'], type=str, help="exclude one or more histo types. options cor for corMatrix, traintest for classifier, roc, sig,eff, sigEff, purEff")
  parser.add_argument("--plotrange", dest="plotrange",nargs='+', default=['none','-1','-1','-1','-1'], type=str, help="preffered histo with x y ranges. Histo options: cor , traintest, roc, sig,eff, sigEff, purEff. eg roc 0 1 0 1")
  parser.add_argument("--xval",dest="xvals",nargs='+',default=['DATA1','DATA2','DATA3'], type=str,help=" extra folder of each k-fold ")
  parser.add_argument("--expSig",dest="expectedSig", default=1.0, type=float,help=" ")
  parser.add_argument("--path",dest="path", default='/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/uboost/CMSSW_10_1_4/src/hep_ml/ScikitWeights_v1', type=str,help=" ")
  parser.add_argument("--inbkgtrain",dest="inbkgtrain", default='train_bkg.root', type=str,help=" ")
  parser.add_argument("--inbkgtest",dest="inbkgtest",nargs='+', default=['test_bkg.root'], type=str,help=" ")
  parser.add_argument("--insigtrain",dest="insigtrain", default='train_sig.root', type=str,help=" ")
  parser.add_argument("--insigtest",dest="insigtest", default='test_sig.root', type=str,help=" ")
  args = parser.parse_args()
  t0 = time.time()
  rangehistos,xmins,xmaxs,ymins,ymaxs = decouple_histo_range(args.plotrange)
  print rangehistos,xmins,xmaxs,ymins,ymaxs
  #read var
  used_columns = """Bprob,BLxy,BeLxy,l1pt,l2pt,kpt,Bcos,Bpt,Beta""".split(",")
  used_columns = [c.strip() for c in used_columns]
  # data load test_sig.root test_bkg.root
  testsignal= root_numpy.root2array(args.insigtest, treename='mytree',branches=used_columns)
  #testbackgr= root_numpy.root2array(args.inbkgtest, treename='mytree',branches=used_columns)
  if "cor" not in args.exclude:
    dataSgn=pd.DataFrame(testsignal)
    dataBkg=pd.DataFrame(testbackgr)
    correlations(dataSgn,name="Sgn")
    correlations(dataBkg,name="Bkg")
  if "traintest" not in args.exclude:
    trainsignal=root_numpy.root2array(args.insigtrain, treename='mytree',branches=used_columns)
    trainbackgr= root_numpy.root2array(args.inbkgtrain, treename='mytree',branches=used_columns)
    trainsignal=root_numpy.rec2array(trainsignal)
    trainbackgr=root_numpy.rec2array(trainbackgr)
    Xtrain=np.concatenate((trainsignal,trainbackgr))
    Ytrain=np.concatenate(([1 for i in range(len(trainsignal))],[0 for i in range(len(trainbackgr))]))
  testsignal=root_numpy.rec2array(testsignal)
  #testbackgr=root_numpy.rec2array(testbackgr)
  #Xtest=np.concatenate((testsignal,testbackgr))
  #Ytest=np.concatenate(([1 for i in range(len(testsignal))],[0 for i in range(len(testbackgr))]))
  precisions=[]; recalls =[]; precisionsrecalls=[]; names=[]; thresholds=[];
  fprs=[]; tprs=[]; decisions=[]; signEffs=[]; effs=[]; signs=[]; count=0;
 
  
  for xval,model,root in zip(args.xvals,args.models,args.inbkgtest): 
    count+=1
    testbackgr= root_numpy.root2array(root, treename='mytree',branches=used_columns)
    testbackgr=root_numpy.rec2array(testbackgr)
    Xtest=np.concatenate((testsignal,testbackgr))
    Ytest=np.concatenate(([1 for i in range(len(testsignal))],[0 for i in range(len(testbackgr))]))
    weightSgn=0;
    for wg in Ytest:
       if wg==1: weightSgn+=1.0
    weightSgn=args.expectedSig/weightSgn
    weightTest=[1 if w==0 else weightSgn for w in Ytest] 
    clf=joblib.load(args.path+"/"+xval+"/"+model+".pkl")
    prob=True
    if 'XGBoost' in model: 
       prob=True;
    if "traintest" not in args.exclude:
      compare_train_test(clf,Xtrain,Ytrain,Xtest,Ytest,bins=30,name=model,xlabel='Classifier Output',ylabel='arbitrary units',grid=True,useProb=prob)
    decision=[]
    if prob:
      decisionTemp=clf.predict_proba(Xtest)
      decision=[x[1] for x in decisionTemp]
    else:
      decision=clf.decision_function(Xtest)    
    print "Results for model ",model
    decisions.append(decision); names.append(str(count))
    if "prEff" not in args.exclude:
      precision, recall, threshold = precision_recall_curve(Ytest,decision,sample_weight=weightTest)
      precisions.append(precision); recalls.append(recall)
      precisionsrecalls.append(precision*recall); 
    if "purEff" not in args.exclude:
      plot_prec_recall_vs_tresh(precision, recall, threshold,name=model)
    fpr,tpr,threshold=roc_curve(Ytest,decision)
    fprs.append(fpr); tprs.append(tpr)
    thresholds.append(threshold); 
    if "sigEff" not in args.exclude:
      signEffs.append(calculate_signEff(Ytest,decision,threshold,sgnWeight=weightSgn))
    if "eff" not in args.exclude:
      effs.append(calculate_efficiency(Ytest,decision,threshold))
    if "sig" not in args.exclude:
      signs.append(calculate_significance(Ytest,decision,threshold,sgnWeight=weightSgn))

  if "purEff" not in args.exclude:
    plot_purities_efficiencies(recalls,precisionsrecalls,names)
  if "roc" not in args.exclude:
    plot_rocs(fprs,tprs,names)
  if "sigEff" not in args.exclude:
    if "sigEff" in rangehistos:
       idx=rangehistos.index('sigEff')
       multi_plot(thresholds,signEffs,names,"sigEff","Classifier Output","Significance*Efficiency",False,False,xmins[idx],xmaxs[idx],ymins[idx],ymaxs[idx])
    else:
       multi_plot(thresholds,signEffs,names,"sigEff","Classifier Output","Significance*Efficiency",False,False,0,1,0,1)  
  if "eff" not in args.exclude:
    multi_plot(thresholds,effs,names,"eff","Classifier Output","Efficiency")
  if "sig" not in args.exclude:
    multi_plot(thresholds,signs,names,"sign","Classifier Output","Significance")

    #plot_precrecs_vs_treshs(precisions,recalls,thresholds,names)
  #for prec,rec in zip(precisions,recalls):
  t1 = time.time()   
  print t1-t0
  
