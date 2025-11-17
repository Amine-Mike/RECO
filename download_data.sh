#!/bin/sh

mkdir -p data
cd data
aria2c -x 16 -s 16 -o speeches.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
cd data
tar xvf speeches.tar.bz2
rm speeches.tar.bz2
