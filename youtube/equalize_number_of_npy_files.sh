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
    real_files=$( find . -name '*.npy' -type f | sed -e 's#./##'| tr ' ' '\n' )
    num_real_files=$(echo $real_files| tr ' ' '\n' | wc -l)
    while (( $I < 110))
    do   
       echo $I
       J=$(($I%$num_real_files)); 
       ln -s data.$J.npy data.$I.npy;
       I=$((I+1));
    done
  fi
done



