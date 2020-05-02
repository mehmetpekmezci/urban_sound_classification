cd downloads
for i in *.m4a *.mp3
do
	f=$(echo $i | sed -e 's/.m4a//'| sed -e 's/.mp3//')
        ffmpeg -i $i -acodec pcm_s16le -ar 44100 $f.wav 
done

