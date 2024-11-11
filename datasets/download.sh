#!/bin/bash

DATA=$1 > /dev/null

if [[ $DATA != "monet2photo" && $DATA != "cezanne2photo" && $DATA != "ukiyoe2photo" && $DATA != "vangogh2photo" ]]; then
    echo "Available datasets are: monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo"
    exit 1
fi

URL=http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/$DATA.zip
ZIP_FILE=./DATA/$DATA.zip
mkdir -p ./DATA
wget -v -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./DATA/
rm $ZIP_FILE