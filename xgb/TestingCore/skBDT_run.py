import os
import pandas as pd
import numpy as np
import root_numpy
import argparse
from sklearn.externals import joblib


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--models", dest="models",nargs='+', default=['adaboost.pkl','gradboost.pkl'], type=str, help="full model name")
  parser.add_argument("--modelnames", dest="modelnames",nargs='+', default=[], type=str, help="name to print of the barcnch for the corresponding TTree")
  parser.add_argument("--extrabrc", dest="extrabranches", default="""Bmass,mll""", type=str, help="branches to copy from measurment TTree.Write all to add tehm all")
  parser.add_argument("--inname",dest="inname", default='measurment_bdt.root', type=str,help="sample to run BDTon")
  parser.add_argument("--outname",dest="outname", default='resultBDT.root', type=str,help=" ")
  parser.add_argument("--modelpath",dest="path", default='/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/uboost/CMSSW_10_1_4/src/hep_ml/ScikitWeights_v1', type=str,help=" ")
   
  parser.add_argument("--replace",dest="replace", default=True, type=bool,help="replace BDT if the output nanme is the same")
  args = parser.parse_args()
 
  if args.replace and os.path.exists(args.outname):
     print "same file exists and the flag is in replace. Replacing..."
     os.remove(args.outname)
   
  modelnames=[]
  if len(args.models)!=len(args.modelnames):
    print "inconsistent # of models with # of names. Adjusting..."
    #del args.modelnames
    for i in range(0,len(args.models)):
      modelnames.append("model_"+str(i+1))
  else:
     for name in args.modelnames:
      modelnames.append(name)
 
  #read var
  used_columns = """Bprob,BLxy,BeLxy,l1pt,l2pt,kpt,Bcos,Bpt,Beta,l1seed""".split(",")
  used_columns = [c.strip() for c in used_columns]
  write_columns = args.extrabranches.split(",")
  if args.extrabranches=='all': write_columns="""Bmass,Bprob,BLxy,BeLxy,l1pt,l2pt,kpt,Bcos,Bpt,Beta,mll,l1eta,l2eta""".split(",")
  #print write_columns
  write_columns = [c.strip() for c in write_columns]
  # data load test_sig.root test_bkg.root
  measure= root_numpy.root2array(args.inname, treename='mytree',branches=used_columns)
  measure=root_numpy.rec2array(measure)
  varsout= root_numpy.root2array(args.inname, treename='mytree',branches=write_columns)
 
  for model,name in zip(args.models,modelnames):
    clf=joblib.load(args.path+"/"+model+".pkl")
    decisionsTemp=clf.predict_proba(measure)
    decisions=[x[1] for x in decisionsTemp]
    decisions=np.array(decisions,dtype=np.float64)
    decisions.dtype=[(name,np.float64)]
    root_numpy.array2root(decisions,args.outname,"mytree")
    root_numpy.array2root(varsout,args.outname,"mytree")
