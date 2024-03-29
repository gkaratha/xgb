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
SAMPLEFR=$1
echo "sample fraction $1 "
shift;
VARSFR=$1
echo "vars fraction $1 "
shift; 
GAMMA=$1
echo "gamma factor $1 "
shift;
PART=$1
echo "part $1"
shift;

TOP="$PWD"

CMSSW_P=/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/uboost/CMSSW_10_1_4/src/
RUNDIR=${CMSSW_P}/hep_ml/TrainingCore
RUNFILE=skBDT_train.py
TESTFILE=skBDT_test.py
MEASUREFILE=skBDT_run.py
DATAPATH=/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/uboost/CMSSW_10_1_4/src/hep_ml/BDTinput/

cd ${CMSSW_P}
eval `scramv1 runtime -sh`
cd ${TOP}
cp ${RUNDIR}/*.py .
cp ${CMSSW_P}/hep_ml/TestingCore/${MEASUREFILE} .

echo train with --clf "'"${CLF}"'" --depth ${DEPTH} --lrate ${SHRINK} --nestima ${NTREE}
MODEL=lRate${SHRINK}_Depth${DEPTH}_Ntrees${NTREE}
if [ "${CLF}" == "adaboost" ]
then
   MODEL=AdaBoost_lRate${SHRINK}_Depth${DEPTH}_Ntrees${NTREE}_SampFraction${SAMPLEFR}
fi
if [ "${CLF}" == "adaboost" ] && [ "${VARSFR}" != "1.0" -o "${GAMMA}" != "1.0" ];
then
   echo "not existing parameters in ada. Skipping not to have duplicates";
   exit 0;
fi

if [ "${CLF}" == "gradboost" ]
then
   MODEL=GradBoost_lRate${SHRINK}_Depth${DEPTH}_Ntrees${NTREE}_SampFraction${SAMPLEFR}_VarFraction${VARSFR}
fi

if [ "${CLF}" == "xgboost" ]
then
   MODEL=XGBoost_lRate${SHRINK}_Depth${DEPTH}_Ntrees${NTREE}_SampFraction${SAMPLEFR}_VarFraction${VARSFR}_Gamma${GAMMA}
fi

echo ${MODEL}

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
   '''re='^[0-9]+$'
   if [[ $PART =~ $re ]]
   then
      if [ $((i + 1)) -gt $((PART)) ] || [  $((i + 1)) -lt $((PART)) ]
      then
        echo "continue..."
        continue
      fi
      FOLDER=${array_name[$i]}_${PART}
   fi'''
   echo " train on ${TRAINBKG} sample"
   echo " test on ${TESTBKG} sample"
   echo " measure on ${MEAS} sample"
   cp ${DATAPATH}/${TRAINBKG} train_bkg.root
   cp ${DATAPATH}/${TESTBKG} test_bkg.root
   cp ${DATAPATH}/${MEAS} measure.root
   echo "training..."
   python skBDT_train.py --clf ${CLF} --depth ${DEPTH} --lrate ${SHRINK} --ntree ${NTREE} --samplefrac ${SAMPLEFR} --varsfrac ${VARSFR} --rgamma ${GAMMA} 
   if [ ! -e ${MODEL}.pkl ]
   then
     echo "weight not found- exiting"
     exit 0;
   fi
   echo "testing..."
   python skBDT_test.py --model ${MODEL} --nevts 2700000000
   mkdir -p /eos/user/g/gkaratha/hep_ml/ScikitWeights/Xval/${FOLDER}
   mkdir -p /eos/user/g/gkaratha/performance/Xval/${FOLDER}
   cp *.pkl /eos/user/g/gkaratha/hep_ml/ScikitWeights/Xval/${FOLDER}
   mkdir ${MODEL}
   mv *.png ${MODEL}
   echo "save weight at ${FOLDER} "
   echo "measuring..."
   python skBDT_run.py --models ${MODEL} --modelnames bdt --inname measure.root --outname ${MODEL}_measure.root --modelpath ${TOP}
   mv ${MODEL}_measure.root ${MODEL}
   mv ${MODEL} /eos/user/g/gkaratha/performance/Xval/${FOLDER}

   echo "finished"
done


