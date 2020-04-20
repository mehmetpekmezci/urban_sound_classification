d=$(basename $(pwd))
echo "Cleaning Saved Model ..."
#sleep 30
echo "rm -Rf ../../save/$d"
rm -Rf ../../save/$d
rm -Rf ../../save/$d-autoencoder
python3 main.py &
