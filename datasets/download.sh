#!/bin/bash

DATA=$1

URL=http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/$DATA.zip
ZIP_FILE=./DATA/$DATA.zip
mkdir -p ./DATA
wget --progress=bar:force -v -N $URL -O $ZIP_FILE