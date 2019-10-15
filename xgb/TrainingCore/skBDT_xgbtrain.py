import sys
import pandas
import numpy as np
from sklearn.cross_validation import train_test_split
from xgboost import XGBClassifier
# this wrapper makes it possible to train on subset of features
from rep.estimators import SklearnClassifier
import root_numpy
import argparse
from sklearn.externals import joblib
from sklearn.utils.class_weight import compute_sample_weight
import time
import os




if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ntree", dest="ntree",default=100, type=int,  help="number of trees")
  parser.add_argument("--depth", dest="depth",default=3, type=int,  help="tree depth")
  parser.add_argument("--lrate", dest="lrate",default=0.3, type=float,  help="learning rate")
  parser.add_argument("--subsample", dest="subsample", default=1.0, type=float, help="fraction of evts")
  parser.add_argument("--gamma", dest="gamma", default=0, type=float, help="fraction of evts")
  parser.add_argument("--nodeweight", dest="nodeweight", default=1.0, type=float, help="fraction of evts")
  parser.add_argument("--scaleweight", dest="scaleweight", default=1.0, type=float, help="fraction of evts")
  parser.add_argument("--lossfunction", dest="lossfunction", default="logistic", type=str, help="fraction of evts")
  parser.add_argument("--modelname", dest="modelname", default="xgbmodel", type=str, help="fraction of evts")
  
  args, unknown = parser.parse_known_args()
  
  for arg in unknown:
      print "warning uknown parameter",arg
 
  print args.depth
  #read var
  used_columns = """Bprob,BLxy,BeLxy,l1pt,l2pt,kpt,Bcos,Bpt,Beta,l1seed""".split(",")
  used_columns = [c.strip() for c in used_columns]

  # data load train_sig.root train_bkg.root
  signal= root_numpy.root2array('train_sig.root', treename='mytree',branches=used_columns)
  backgr= root_numpy.root2array('train_bkg.root', treename='mytree',branches=used_columns)
  signal=root_numpy.rec2array(signal)
  backgr=root_numpy.rec2array(backgr)

  print'train on', used_columns
  model=args.modelname
   
  print "trainning Model ",model
  
  print "using hyperparams"
  for arg in vars(args):
      print "hyperparameter",arg,getattr(args, arg)

  X=np.concatenate((signal,backgr))
  Y=np.concatenate(([1 for i in range(len(signal))],[0 for i in range(len(backgr))]))
  X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.05,random_state=42)


  #model definition
  weightTrain= compute_sample_weight(class_weight='balanced', y=Y_train)
  weightTest= compute_sample_weight(class_weight='balanced', y=Y_test)
  bdt=XGBClassifier(max_depth=args.depth,n_estimators=args.ntree,learning_rate=args.lrate, min_child_weight=args.nodeweight, gamma=args.gamma, subsample=args.subsample, scale_pos_weight=args.scaleweight, objective= 'binary:'+args.lossfunction) 

  #training
  start = time.clock()
  bdt.fit(X_train,Y_train,sample_weight=weightTrain)
  elapsed = time.clock()
  elapsed = elapsed - start

  #save weight
  print "Time spent is: ", elapsed,"saving model"
  joblib.dump(bdt,model+'.pkl')
 
    
