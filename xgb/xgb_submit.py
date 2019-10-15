import os, subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntree", dest="ntree", nargs='+',default=["none"], type=str,    help="number of trees")
    parser.add_argument("--depth", dest="depth", nargs='+',default=["none"], type=str,    help="tree levels")
    parser.add_argument("--lrate", dest="lrate", nargs='+',default=["none"], type=str,    help="learning rate")
    parser.add_argument("--subsample", dest="subsample", nargs='+',default=["none"], type=str,    help="subsample to use")
    parser.add_argument("--gamma", dest="gamma", nargs='+',default=["none"], type=str, help="gamma factor")
    parser.add_argument("--nodeweight", dest="nodeweight", nargs='+',default=["none"], type=str,    help="min #evts in node")
    parser.add_argument("--scaleweight", dest="scaleweight", nargs='+',default=["none"], type=str,    help="")
    parser.add_argument("--lossfunction", dest="lossfunction", nargs='+',default=["none"], type=str,    help="loss function for node cuts")
    parser.add_argument("-q", dest="flavor", default="workday", type=str,    help="queue to submit")
    parser.add_argument("--eek", dest="eeK", action='store_true', default=False,   help="eeK flag")
    args = parser.parse_args()
    print args


for tree in args.ntree:
  for lev in args.depth:
    for rate in args.lrate:
      for subsamp in args.subsample:
        for nweight in args.nodeweight:
          for gam in args.gamma:  
            for scweight in args.scaleweight:
             for lossf in args.lossfunction:
               name="xgboost_ntree_"+tree+"_depth_"+lev+"_lrate_"+rate+"_subsample_"+subsamp+"_nodeweight_"+nweight+"_gamma_"+gam+"_scaleweight_"+scweight+"_lossfunction_"+lossf
               line="universe = vanilla\n"
               line+="executable = SubmitCore/batch_grid_Exgb.sh\n"
               with open("condor_temp.sub",'w') as out:
                 line+="arguments = {ntree} {depth} {lrate} {subsample} {gamma} {nodeweight} {scaleweight} {lossfunction} {eeK} \n".format(
                    ntree="--ntree "+tree if tree !="none" else "",  
                    depth="--depth "+lev if lev !="none" else "", 
                    lrate="--lrate "+rate if rate!="none" else "", 
                    subsample="--subsample "+subsamp if subsamp!="none" else "",
                    gamma="--gamma "+gam if gam!="none" else "", 
                    nodeweight="--nodeweight "+nweight if nweight!="none" else "", 
                    scaleweight="--scaleweight "+scweight if scweight!="none" else "", 
                    lossfunction="--lossfunction "+lossf if lossf!="none" else "",
                    eeK="eeK" if args.eeK else ""
                 )
                 line+='output = grid_output/bdt_{job}.out\n'.format(job=name)
                 line+='error = grid_error/bdt_{job}.err\n'.format(job=name)
                 line+='log = grid_log/bdt_{job}.log\n'.format(job=name)
                 line+='transfer_output_files   = ""\n'                 
                 line+='+JobFlavour = \"{flavor}\" \n'.format(flavor=args.flavor)
                 line+="queue\n"
                 out.write(line);
               out.close()
               print "submitting "+name+" tree"
               os.system('condor_submit condor_temp.sub')
               subprocess.check_output(['rm',"condor_temp.sub"])
               print "done"
