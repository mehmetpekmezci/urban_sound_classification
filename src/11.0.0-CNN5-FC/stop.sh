if [ -e run.pid ]
then
  PID=$(cat run.pid)
  if [ "$PID" != "" ]
  then	  
    echo "Stopping Process with PID=$PID :"
    ps -ef | grep "$PID" | head -1
    kill -9 $PID
    sleep 2
    ps -ef | grep "$PID" | head -1
  else
    echo "No such process with PID=$PID"
  fi
else
  echo "No 'run.pid' file found"
fi

rm -f run.pid
