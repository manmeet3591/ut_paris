#!/bin/bash

for file in graphcast_2017_*
do
echo $file
python extract_graphcast_paris.py $file
done

for file in graphcast_2018_12_*
do
echo $file
python extract_graphcast_paris.py $file
done

for file in graphcast_2019_*
do
echo $file
python extract_graphcast_paris.py $file
done

for file in graphcast_2020_*
do
echo $file
python extract_graphcast_paris.py $file
done

for file in graphcast_2021_*
do
echo $file
python extract_graphcast_paris.py $file
done


for file in graphcast_2022_*
do
echo $file
python extract_graphcast_paris.py $file
done

for file in graphcast_2023_*
do
echo $file
python extract_graphcast_paris.py $file
done
