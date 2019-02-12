d=$(dirname $(pwd))
echo "Cleaning Saved Model ..."
#sleep 30
rm -Rf ../../save/$d
python3 main.py &
