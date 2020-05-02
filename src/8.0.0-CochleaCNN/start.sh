d=$(basename $(pwd))
echo "Clean Model ? (y/n) :"
read answer
if [ "$answer" = "y" ]
then
  echo "rm -Rf ../../save/$d"
#  rm -Rf ../../save/$d
else
  echo "Just continuing to train deep net. "
fi 
python3 main.py &
echo $! > run.pid

