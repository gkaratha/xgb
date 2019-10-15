#!/bin/bash

CLF=$1
echo "clf $1 "
shift;
NTREE=$1
echo "tree $1 "
shift;
DEPTH=$1
echo "depth $1 "
shift;
SHRINK=$1
echo "shrink $1 "
shift;
NODESIZE=$1
echo "node size $1 "
shift;
CUTS=$1
echo "cuts $1 "
shift; 

TOP="$PWD"

CMSSW_P=/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/uboost/CMSSW_10_1_4/src/
RUNDIR=${CMSSW_P}/hep_ml/TrainingCore
RUNFILE=skBDT_train.py
TESTFILE=skBDT_train.py
MEASUREFILE=skBDT_run.py
DATAPATH=/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/uboost/CMSSW_10_1_4/src/hep_ml/BDTinput/

TMVA=true

if [ ${TMVA} ]
then
  RUNDIR=${CMSSW_P}/hep_ml/TMVAcode
  RUNFILE=BDT_train_grid.C
  TESTFILE=BDT_test.C
  MEASUREFILE=BDT_measure.C
fi

cd ${CMSSW_P}
eval `scramv1 runtime -sh`
cd ${TOP}
cp ${RUNDIR}/* .

echo train with --clf "'"${CLF}"'" --depth ${DEPTH} --lrate ${SHRINK} --nestima ${NTREE}
MODEL=lRate${SHRINK}_Depth${DEPTH}_Ntrees${NTREE}
if [ "${CLF}" == "adaboost" ]
then
   MODEL=AdaBoost_lRate${SHRINK}_Depth${DEPTH}_Ntrees${NTREE}_NodeSize${NODESIZE}_Cuts${CUTS}
fi

if [ "${CLF}" == "gradboost" ]
then
   MODEL=GradBoost_lRate${SHRINK}_Depth${DEPTH}_Ntrees${NTREE}_NodeSize${NODESIZE}_Cuts${CUTS}
fi

echo ${MODEL}

array_train=("trainBkg_bdt_xvalpart_sideBands_lowQ_part1.root" "trainBkg_bdt_xvalpart_sideBands_lowQ_part2.root" "trainBkg_bdt_xvalpart_sideBands_lowQ_part3.root")
array_test=("trainBkg_bdt_xvalpart_sideBands_lowQ_part2.root" "trainBkg_bdt_xvalpart_sideBands_lowQ_part3.root" "trainBkg_bdt_xvalpart_sideBands_lowQ_part1.root")
array_meas=("measurment_bdt_xvalpart_part3.root" "measurment_bdt_xvalpart_part1.root" "measurment_bdt_xvalpart_part2.root")
array_name=("DATA3" "DATA1" "DATA2")

cp ${DATAPATH}/train_sig.root .
cp ${DATAPATH}/test_sig.root .


length=${#array_train[@]}
for (( i=0; i<$length; i++ ))
do
   TRAINBKG=${array_train[$i]}
   TESTBKG=${array_test[$i]}
   MEAS=${array_meas[$i]}  
   FOLDER=${array_name[$i]}
   echo " train on ${TRAINBKG} sample"
   echo " test on ${TESTBKG} sample"
   echo " measure on ${MEAS} sample"
   cp ${DATAPATH}/${TRAINBKG} train_bkg.root
   cp ${DATAPATH}/${TESTBKG} test_bkg.root
   cp ${DATAPATH}/${MEAS} measure.root
   echo "training..."
   if [ ${TMVA} ]
   then
     root -l -q -b 'BDT_train_grid.C("'${CLF}'","'${NTREE}'","'${DEPTH}'","'${SHRINK}'","'${NODESIZE}'","'${CUTS}'")'
   else 
     python skBDT_train.py --clf ${CLF} --depth ${DEPTH} --lrate ${SHRINK} --ntree ${NTREE} --samplefrac ${SAMPLEFR} --varsfrac ${VARSFR} --rgamma ${GAMMA}
   fi

   echo "testing..."
   if [ ${TMVA} ] 
   then
     root -l -q -b 'BDT_test.C("'${MODEL}'",6.0)'
   else
      python skBDT_test.py --model ${MODEL} --expSig 6.0
   fi
   mkdir -p ${CMSSW_P}/hep_ml/TMVAWeights/Xval/${FOLDER}
   mkdir -p /eos/user/g/gkaratha/performance_tmva/Xval/${FOLDER}
   
   cp dataset3/weights/*.xml ${CMSSW_P}/hep_ml/TMVAWeights/Xval/${FOLDER}
   mkdir ${MODEL}
  
   mv FOM_${MODEL}.root ${MODEL}
   echo "measuring..."
   if [ ${TMVA} ]
   then
      root -l -q -b 'BDT_measure.C("'${MODEL}'",false)'
   else
      python skBDT_run.py --models ${MODEL} --modelnames bdt --inname measure.root --outname ${MODEL}_measure.root --modelpath ${TOP}
   fi
   mv ${MODEL}_measure.root ${MODEL}
   mv ${MODEL} /eos/user/g/gkaratha/performance/TMVA/Xval/${FOLDER}

   echo "finished"
done


