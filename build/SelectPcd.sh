#!/bin/bash
# sh SelectPcd.sh obj_name start_frame_num
# for selecting type '1', set the start frame to write
count="$2"
yes='y'
for f in *.pcd
do
	echo "$f";
	pcd_viewer "$f";
	read select;
	echo "$select";
	if [ "$select" -eq 1 ]; then
		echo "select file";
		count=`expr $count + 1`;
		echo "$count";
		mv "$f" "$1"-"$count".pcd;
	fi
done
