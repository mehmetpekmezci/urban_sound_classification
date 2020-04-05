#!/bin/bash

PATH=$PATH:/usr/local/bin

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
           #youtube-dl -c -r 100k --extract-audio --audio-format m4a --output  downloads/$name.m4a $url
           youtube-dl -c -r 900k --extract-audio --audio-format m4a --output  downloads/$name.m4a $url
	fi
  done

done
