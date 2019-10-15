import os, subprocess, sys
import re
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from optparse import OptionParser

def canvas3plot(valX,valY1,valY2,valY3,title,labelX,labelY):
  plt.figure(1)
  line1,=plt.plot(valX,valY1,label='ada',color='blue',linewidth=3) 
  line2,=plt.plot(valX,valY2,label='grad',color='red',linewidth=3)
  line3,=plt.plot(valX,valY3,label='xgb',color='black',linewidth=3)
  plt.legend(handles=[line1,line2,line3],loc='best')
  plt.xlabel(labelX,fontsize=20)
  plt.ylabel(labelY,fontsize=20)
  plt.title(title,fontsize=20,weight='bold')
  #plt.show()
  plt.savefig(labelX+".png")
  plt.close()

def meanPerValue(values,models,index):
   num=[]
   for value in values:
      temp=0; den=0;
      for model in models:
         #print model[-1]
         if value==model[index]:
            temp+=model[-1]
            den+=1.0
      num.append(temp/den)
   return num

def wordlist(line,keyword):
  words=[]
  if keyword in line:
    words=line.split() 
  return words

def wordindex(line,keyword,plus):
  words=wordlist(line,keyword)
  c=-1; result=-1
  for word in words:
    c+=1;
    if word==keyword:
      result=c
  if result>-1:
     return words[result+plus]
  else:
     return None

 
if __name__ == "__main__":  
  parser = OptionParser()
  parser.add_option("-p", dest="mypath", default='grid_output', type="str", help=" folder ")
  parser.add_option("-m", dest="metric", default="sig*eff", type="str", help="options: sig, pur*eff, sig*eff")
  (options, args) = parser.parse_args()
  if options.deletefraction>0:
     print "Warning ",options.deletefraction," of worse model weights will be erased; folder ",options.deleteweightsdir
  if options.metric=="sig*eff":
     metric="significance*efficiency"
  if options.metric=="sig":
     metric="significance"
  if options.metric=="pur*eff":
     metric="eff*pur="
  lsfs=[]
  for dirname, dirnames, filenames in os.walk(options.mypath):
   for filename in filenames:
     lsfs.append(filename)
  models_ada=[]; models_grad=[]; models_xgb=[]
  xgb_hparams=[]; ada_hparams=[]; grad_hparams=[];
  for lsf in lsfs:
   with open(options.mypath+"/"+lsf,'r') as fh:
     lines = fh.readlines()
     perform_temp=[]; ntree=0; depth=0; rate=0
     #print len(lines),lsf
     ntree=wordindex(lines[1],"tree",1)
     depth=wordindex(lines[2],"depth",1)
     rate=wordindex(lines[3],"shrink",1)
     #print lines[1],ntree
     params=[float(ntree),float(depth),float(rate)]
     for line in lines:
       #depth=wordindex(line,"depth",1)
       #rate=wordindex(line,"shrink",1)
       iword=wordindex(line,"Model",1)
       
       if iword is not None:
         perform_temp.append(iword)
       iword=wordindex(line,metric,1)
       if iword is not None:
         perform_temp.append(float(iword))
     #if len(perform_temp)==1: print perform_temp
     if len(perform_temp)>1 and "GradBoost" in perform_temp[0]:
         #print params
         models_grad.append(perform_temp); params.append(perform_temp[1]); 
         grad_hparams.append(params);
     if len(perform_temp)>1 and "AdaBoost" in perform_temp[0]:
         models_ada.append(perform_temp);  params.append(perform_temp[1]); 
         ada_hparams.append(params);     
     if len(perform_temp)>1 and "XGBoost" in perform_temp[0]:
         models_xgb.append(perform_temp);  params.append(perform_temp[1]); 
         xgb_hparams.append(params); 
         
  models_ada.sort(key=lambda g: g[1])
  models_grad.sort(key=lambda g: g[1])
  models_xgb.sort(key=lambda g: g[1])
  print models_ada[-1*options.top:]
  print models_grad[-1*options.top:]
  print models_xgb[-1*options.top:]
  value_tree=[100,500,1000]; value_depth=[3,5,7]; value_rate=[0.5,0.1,0.01];
  mean_tree_ada=meanPerValue(value_tree,ada_hparams,0)
  mean_depth_ada=meanPerValue(value_depth,ada_hparams,1)
  mean_rate_ada=meanPerValue(value_rate,ada_hparams,2)
  mean_tree_grad=meanPerValue(value_tree,grad_hparams,0)
  mean_depth_grad=meanPerValue(value_depth,grad_hparams,1)
  mean_rate_grad=meanPerValue(value_rate,grad_hparams,2)
  mean_tree_xgb=meanPerValue(value_tree,xgb_hparams,0)
  mean_depth_xgb=meanPerValue(value_depth,xgb_hparams,1)
  mean_rate_xgb=meanPerValue(value_rate,xgb_hparams,2)
  canvas3plot(value_tree,mean_tree_ada,mean_tree_grad,mean_tree_xgb,"CMS Preliminary","Ntree",options.metric)
  canvas3plot(value_depth,mean_depth_ada,mean_depth_grad,mean_depth_xgb,"CMS Preliminary","Depth",options.metric)
  canvas3plot(value_rate,mean_rate_ada,mean_rate_grad,mean_rate_xgb,"CMS Preliminary","Rlearning",options.metric)
  

