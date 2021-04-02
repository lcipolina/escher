#!/bin/sh

cd scripts

for d in "1 2" "2 3" "3 4" "4 5";
do
	mkdir "../workspace-$d"
	cd "../workspace-$d"
	python3 ../scripts/code.py 20 0.5 $d
done
