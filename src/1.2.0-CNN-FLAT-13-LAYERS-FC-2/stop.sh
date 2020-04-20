kill -9  $(ps -ef | grep python| grep main.py| awk '{ print $2}')
