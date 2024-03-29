#!/bin/bash
#parse optional values
VARS=""
MODEL="XGBoost"
for i in "$@" 
do
  echo "$i"
  VARS+=" $i"
  MODEL+="_${i//-}"
  shift
done

echo ${MODEL}


TOP="$PWD"

CMSSW_P=/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/uboost/CMSSW_10_1_4/src/
RUNDIR=${CMSSW_P}/hep_ml/TrainingCore
TRAINFILE=skBDT_xgbtrain.py
TESTFILE=skBDT_xgbtest.py
MEASUREFILE=skBDT_run.py
DATAPATH=/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/uboost/CMSSW_10_1_4/src/hep_ml/BDTinput/

cd ${CMSSW_P}
eval `scramv1 runtime -sh`
cd ${TOP}
cp ${RUNDIR}/${TRAINFILE} .
cp ${RUNDIR}/${TESTFILE} .
cp ${RUNDIR}/plotting_variables.py .
cp ${CMSSW_P}/hep_ml/TestingCore/${MEASUREFILE} .

array_train=("trainBkg_bdt_xvalpart_sideBands_lowQ_part1.root" "trainBkg_bdt_xvalpart_sideBands_lowQ_part2.root" "trainBkg_bdt_xvalpart_sideBands_lowQ_part3.root")
array_test=("trainBkg_bdt_xvalpart_sideBands_lowQ_part2.root" "trainBkg_bdt_xvalpart_sideBands_lowQ_part3.root" "trainBkg_bdt_xvalpart_sideBands_lowQ_part1.root")
array_meas=("measurment_bdt_xvalpart_part3.root" "measurment_bdt_xvalpart_part1.root" "measurment_bdt_xvalpart_part2.root")
array_name=("DATA_train1_meas3" "DATA_train2_meas1" "DATA_train3_meas2")

cp ${DATAPATH}/train_sig.root .
cp ${DATAPATH}/test_sig.root .


length=${#array_train[@]}
for (( i=0; i<$length; i++ ))
do
   TRAINBKG=${array_train[$i]}
   TESTBKG=${array_test[$i]}
   MEAS=${array_meas[$i]}  
   FOLDER=${array_name[$i]}
   echo part $((i+1))
   echo " train on ${TRAINBKG} sample"
   echo " test on ${TESTBKG} sample"
   echo " measure on ${MEAS} sample"
   cp ${DATAPATH}/${TRAINBKG} train_bkg.root
   cp ${DATAPATH}/${TESTBKG} test_bkg.root
   cp ${DATAPATH}/${MEAS} measure.root
   echo "training..."
   python ${TRAINFILE} ${VARS} --modelname ${MODEL}
   if [ ! -e ${MODEL}.pkl ]
   then
     echo "weight not found- exiting"
     exit 0;
   fi
   echo "testing..."
   python ${TESTFILE} --model ${MODEL} --nevts 2700000000
   mkdir -p /eos/user/g/gkaratha/XGBWeights/Xval/${FOLDER}
   mkdir -p /eos/user/g/gkaratha/performance/Xval/${FOLDER}
   cp *.pkl /eos/user/g/gkaratha/XGBWeights/Xval/${FOLDER}
   mkdir ${MODEL}
   mv *.png ${MODEL}
   echo "save weight at XGBWeights/Xval/${FOLDER} "
   echo "measuring..."
   python skBDT_run.py --models ${MODEL} --modelnames bdt --inname measure.root --outname ${MODEL}_measure.root --modelpath ${TOP}
   mv ${MODEL}_measure.root ${MODEL}
   mv ${MODEL} /eos/user/g/gkaratha/performance/Xval/${FOLDER}

   echo "finished"
done
