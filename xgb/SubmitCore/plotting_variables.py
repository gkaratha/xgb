import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas.core.common as com
from pandas.core.index import Index
from pandas.tools import plotting
from pandas.tools.plotting import scatter_matrix
import numpy as np
from root_numpy import root2array, rec2array
import pandas as pd
from sklearn.metrics import roc_curve, auc
from root_numpy import array2root


def plot_signal_background(signal,bkg,var=None,grid=True,log=False,bins=10,norm=True,xlabel=None,legSgn='sgn',legBkg='bkg',xmin=None,xmax=None):
   i=-1   
   if var is not None:
      if xlabel is not None:
         if len(xlabel)!=len(var): print 'different # of vars and labels'
      if xmin is not None:
         if len(xmin)!=len(var): print 'different # of vars and xmin'
      if xmax is not None:
         if len(xmax)!=len(var): print 'different # of vars and xmax'
   for plot in var:
      i+=1
      #plt.figure()
      #plt.show()
      #signal.loc[:,plot]*=1.0/len(signal)
      if xmin is None:
         low = min(signal[plot].min(),bkg[plot].min())
      else:
         low=xmin[i]
      if xmax is None:
          high = max(signal[plot].max(),bkg[plot].max())
      else:
          high=xmax[i]
      #print high
      if norm :
          weiS = np.repeat(1.0/len(signal), len(signal))
          weiB = np.repeat(1.0/len(bkg), len(bkg))
      else:
          weiS = np.repeat(1.0, len(signal))
          weiB = np.repeat(1.0, len(bkg))
      #print(wei)
      ax1=signal.hist(weights=weiS,column=plot,range=(low,high),label=legSgn,sharex=True,histtype='step',log=log)
      bkg.hist(weights=weiB,column=plot,label=legBkg,range=(low,high),color='red',ax=ax1,histtype='step',log=log)     
      if norm:
             plt.ylim([0,1.2])
      plt.title('CMS Preliminary')
      if xlabel is None:
             plt.xlabel(plot)
      else:
             plt.xlabel(xlabel[i])   
      plt.ylabel('density')
      plt.legend(loc="upper right")
     
      #plt.show()
      plt.savefig("/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/uboost/CMSSW_10_1_4/src/hep_ml/1dplot_"+plot+".png")
      plt.close()
      
def scatter_plot(data,varX,varY,legX='var1',legY='var2',title='CMS Preliminary'):
    dataSgn.plot.scatter(x=varX,y=varY,c='y')
    plt.ylabel(legY)
    plt.xlabel(legX)
    plt.title(title)
    if title is 'CMS Preliminary':
       plt.savefig("/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/uboost/CMSSW_10_1_4/src/hep_ml/d2plot_"+varX+"vs"+varY+".png")
    else:
        plt.savefig("/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/uboost/CMSSW_10_1_4/src/hep_ml/d2plot_"+varX+"vs"+varY+"_"+title+".png")
    #plt.show()

def simple_plot(Xdata,Ydata,varX,varY,name,logX=False,logY=False):
    plt.plot(Xdata,Ydata,linewidth=3.0)
    plt.rcParams.update({'font.size': 22})
    plt.ylabel(varY)
    plt.xlabel(varX)
    plt.title('CMS Preliminary')
    if logY:
      plt.semilogy()
    if logX:
      plt.semilogx()
    if logY and logX:
      plt.loglog()
    plt.savefig(name+".png")
    plt.show()

def correlations(data,title='CMS Preliminary', **kwds):
  """Calculate pairwise correlation between features.
  Extra arguments are passed on to DataFrame.corr()
  """
  # simply call df.corr() to get a table of
  # correlation valu`es if you do not need
  # the fancy plotting
  corrmat = data.corr(**kwds)
  print corrmat
  fig, ax1 = plt.subplots(ncols=1, figsize=(6,5))
  opts = {'cmap': plt.get_cmap("RdBu"),
  'vmin': -1, 'vmax': +1}
  heatmap1 = ax1.pcolor(corrmat, **opts)
  plt.colorbar(heatmap1, ax=ax1)
  ax1.set_title("Correlations")
  labels = corrmat.columns.values
  for ax in (ax1,):
  # shift location of ticks to center of the bins
    ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
    ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
    ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
    ax.set_yticklabels(labels, minor=False)
  plt.tight_layout()
  plt.title(title)
  if title is not 'CMS Preliminary':
    plt.savefig("/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/uboost/CMSSW_10_1_4/src/hep_ml/CorMatrix_"+title+".png")
  else:
    plt.savefig("/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/uboost/CMSSW_10_1_4/src/hep_ml/CorMatrix.png")

def plot_roc(fpr,tpr,roc_auc,title='CMS Preliminary',name="test_bdt"):
   print "ROC ",roc_auc
   plt.plot(fpr,tpr,lw=1,label='ROC (area %0.2f)'%(roc_auc))
   plt.plot([0,1],[0,1], "--", color=(0.6,0.6,0.6),label="random")
   #plt.xlim([-0.05,1.05])
   #plt.ylim([-0.05,1.05])
   plt.rcParams.update({'font.size': 22})
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title(title)
   plt.legend(loc="lower right")
   plt.grid()
   plt.savefig("ROC_"+name+".png")
   plt.semilogx()
   plt.savefig("logxROC_"+name+".png")

   plt.close()
   #plt.show()

def plot_purity_efficiency(rec,rec_prec,name="test_model",xlabel='efficiency',ylabel='purity*efficinency',title='CMS Preliminary',findmax=True):
   if findmax:
      ymax=max(rec_prec)
      xmax=rec[rec_prec.argmax()]
      print "Max eff*pur=",ymax," eff=",xmax
      plt.plot(rec,rec_prec,label='p*e (max %0.2f )'%(ymax*100)+"%",linewidth=3)
   else:
      plt.plot(rec,rec_prec,linewidth=3)
   plt.xlabel(xlabel)
   plt.ylabel(ylabel)
   plt.rcParams.update({'font.size': 22})
   plt.xlim([-0.05,1.05])
   plt.ylim([-0.05,1.05])
   plt.title(title)
   plt.legend(loc="best")
   plt.grid()
   plt.savefig("purity_eff_"+name+".png")
   plt.close()
#   plt.show()

def compare_purity_efficiency(rec=[],rec_prec=[],name=[],xlabel='efficiency',ylabel='purity*efficinency',title='CMS Preliminary',findmax=True):
   for x in xrange(0,len(rec)):
      plt.plot(rec[x],rec_prec[x],lw=1,label=name[x])
   plt.xlabel(xlabel)
   plt.ylabel(ylabel)
   plt.title(title)
   plt.legend(loc="best")
   plt.grid()
   plt.savefig("/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/uboost/CMSSW_10_1_4/src/hep_ml/compare_purity_eff.png")
   plt.close()


def plot_prec_recall_vs_tresh(precisions, recalls, thresholds,xlabel='Classifier output',ylabel='arbitrary units',title='CMS Preliminary',name="test_class"):
    plt.plot(thresholds, precisions[:-1], 'b', label='purity',linewidth=3.0)
    plt.plot(thresholds, recalls[:-1], 'r', label = 'efficiency',linewidth=3.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.rcParams.update({'font.size': 22})
    plt.title(title)
    plt.legend(loc='best')
    plt.ylim([-0.01,1.05])
    plt.savefig("prec_rec_clf_output_"+name+".png")
    plt.close()
    #plt.show()

def plot_classifier_test(decisionsTrainS,decisionsTrainB,decisionsTestS,decisionsTestB,bins=30,name='test',xlabel='Classifier Output',ylabel='arbitrary units',grid=True):
    low = min(min(decisionsTrainS),min(decisionsTrainB),min(decisionsTestS),min(decisionsTestB))
    high = max(max(decisionsTrainS),max(decisionsTrainB),max(decisionsTestS),max(decisionsTestB))
    plt.hist(decisionsTrainS,color='blue',alpha=0.5,range=(low,high),bins=bins,histtype='stepfilled',normed=True,label='Sgn(train)')
    plt.hist(decisionsTrainB,color='red',alpha=0.5,range=(low,high),bins=bins,histtype='stepfilled',normed=True,label='Bkg(train)')
    plt.title('CMS Preliminary')
    hist, bins = np.histogram(decisionsTestS,bins=bins, range=(low,high), normed=True)
    scale = len(decisionsTestS) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='blue', label='Sgn(test)')
    hist, bins = np.histogram(decisionsTestB,bins=bins, range=(low,high), normed=True)
    scale = len(decisionsTestB) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    plt.errorbar(center, hist, yerr=err, fmt='o', c='red', label='Bkg(test)')
    plt.legend(loc="upper center")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.rcParams.update({'font.size': 22})
    plt.savefig("test_train_"+name+".png")
    if grid: plt.grid()
    plt.tight_layout()
    plt.close()
    #plt.show()
    plt.close()
    
  
   


  #plt.show()

#plot_signal_background(dataSgn, dataBkg,
#     var=["Bprob","l1pt","Beta","l2pt","kpt","BLxy","BeLxy","Bcos"],bins=20,xlabel=['Prob','pT(l1)',"eta(B)","pT(l2)","pT(K)","Lxy(B)","sigma Lxy","cos(alpha)"],xmin=[0,0,-2.5,0,0,0,0,-1],xmax=[0.5,20,2.5,15,15,1,0.1,1])
#scatter_plot(dataBkg,"l1pt","l2pt","pT(l1)","pT(l2)",title="signal")
#correlations(dataSgn,title="signal")
#correlations(dataBkg,title="background")

