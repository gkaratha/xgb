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

TOP="$PWD"

CMSSW_P=/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/uboost/CMSSW_10_1_4/src/
RUNDIR=${CMSSW_P}/hep_ml/TrainingCore
RUNFILE=uBoost_train.py
TESTFILE=uBoost_test.py

TRAINPATH=/afs/cern.ch/work/g/gkaratha/private/SUSYCMG/HLT/efficiency/Analizer/MiniAOD_10_1_4/lite/CMSSW_10_2_4/src/HLTAnalysis/TriggerAnalyzer/python
TRAINSIG=trainSgn_bdt_LowQ.root
TRAINBKG=trainBkg_bdt_sideBands_lowQ.root
TESTSIG=testSgn_bdt_LowQ.root
TESTBKG=test_bdt_sideBands_lowQ.root

cd ${CMSSW_P}
eval `scramv1 runtime -sh`
cd ${TOP}
cp ${RUNDIR}/*.py .

cp ${TRAINPATH}/${TRAINSIG} train_sig.root
cp ${TRAINPATH}/${TRAINBKG} train_bkg.root

cp ${TRAINPATH}/${TESTSIG} test_sig.root
cp ${TRAINPATH}/${TESTBKG} test_bkg.root

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

python uBoost_train.py --clf ${CLF} --depth ${DEPTH} --lrate ${SHRINK} --ntree ${NTREE} --samplefrac ${SAMPLEFR} --varsfrac ${VARSFR} --rgamma ${GAMMA} 
if [ ! -e ${MODEL}.pkl ]
then
   echo "weight not found- exiting"
   exit 0;
fi
echo "koyrades "

python uBoost_test.py --model ${MODEL} --expSig 6.0
cp *.pkl ${CMSSW_P}/hep_ml/ScikitWeights
mkdir ${MODEL}
cp *.png ${MODEL}

cp -r ${MODEL} ${CMSSW_P}/hep_ml/performance
