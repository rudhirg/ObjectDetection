#!/bin/bash

#set folder = './water_bottle_1'
#'\home\rudhir\dataset\rgbd-dataset\water_bottle\water_bottle_1'
cd "$1"
for f in *
do
#	data = 'identify *_crop.png | grep -o "[[:digit:]]*x[[:digit:]]*+" | sed "s/[0-9]*x/1 0 0 &/"'
	if [ "`expr \"$f\" : \".*depth.png\"`" = "0" ];
	then
		dir="$1";
		echo "$1/$f";
	fi
done
