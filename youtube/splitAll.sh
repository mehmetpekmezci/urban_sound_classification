cd downloads
for i in *.wav
do
	f=$(echo $i| sed -e 's/.wav//')
	mkdir $f
        ffmpeg -i $i -f segment -segment_time 4 -c copy $f/part%05d.wav
done

