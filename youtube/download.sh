#!/bin/bash

mkdir -p downloads


Running=true

while $Running
do
  Running=false
  for i in $(cat list.txt)
  do
	name=$(echo $i | sed -e 's/##.*//')
	url=$(echo $i | sed -e 's/.*##//')
	if [ ! -f downloads/$name.m4a ]
	then
	   Running=true
	   echo "downloading $name"
           youtube-dl --extract-audio --audio-format m4a --output  downloads/$name.m4a $url
	fi
  done

done
