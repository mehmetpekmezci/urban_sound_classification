cd raw.org
YOUTUBE_DATA_DIR=$(pwd)

for d in *
do
  if [ -d $YOUTUBE_DATA_DIR/$d ]
  then
    echo "Entering into directory $YOUTUBE_DATA_DIR/$d"
    cd $YOUTUBE_DATA_DIR/$d
    N=$(ls *.npy| wc -l)
    I=$N
    J=0
    while (( $I < 110))
    do   
       echo $I
       ln -s data.$J.npy data.$I.npy;
       J=$((J+1)); 
       I=$((I+1));
    done
  fi
done

