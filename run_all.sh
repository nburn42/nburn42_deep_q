for f in *.py; do
    gnome-terminal -e "python $f" &
done
