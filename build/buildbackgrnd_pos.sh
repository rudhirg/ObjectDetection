#!/bin/bash

cd "$1"
for f in *
do
	if [ "`expr \"$f\" : \".*_crop.png\"`" != "0" ];
	then
		dir="$1"
		echo "$1/$f";
	fi
done

