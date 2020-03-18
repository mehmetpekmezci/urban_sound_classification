d=$(basename $(pwd))
continue_training=""
echo "Clean Model ? (y/n) :"
read answer
if [ "$answer" = "y" ]
then
  echo "rm -Rf ../../save/$d"
  rm -Rf ../../save/$d
else
  echo "Continue Training ? (y/n) :"
  read answer
  if [ "$answer" = "y" ]
  then
    echo "Just continuing to train deep net. "
    continue_training="continue_training"
  else
    echo "Will not train AutoEncoder"
  fi
fi 
python3 main.py $continue_training&
echo $! > run.pid

