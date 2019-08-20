cd downloads
for i in *.m4a
do
	f=$(echo $i | sed -e 's/.m4a//')
        ffmpeg -i $i -acodec pcm_s16le -ar 22050 $f.wav 
done

