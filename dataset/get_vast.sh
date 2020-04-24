#!/bin/bash

if [[ `pwd` == *dataset ]]
then
    WS="./VAST/raw"
else
    WS="./dataset/VAST/raw"
fi

mkdir -p ${WS}
cd $WS
curl -o "train.zip" "http://thevastproject.inria.fr/VASTDatabase/MITKemar/VASTTraining/VASTTraining.zip"
curl -o "val.zip" "http://thevastproject.inria.fr/VASTDatabase/MITKemar/VASTTesting/VASTTest1/VASTTestingSet1.zip"
curl -o "test.zip" "http://thevastproject.inria.fr/VASTDatabase/MITKemar/VASTTesting/VASTTest2/VASTTestingSet2.zip"
curl -o "anechoic.mat" "http://thevastproject.inria.fr/VASTDatabase/MITKemar/VASTTraining/Rooms/TrainRoom0.mat"
unzip train.zip -d train/
unzip val.zip -d val/
unzip test.zip -d test/

rm train.zip
rm test.zip
rm val.zip