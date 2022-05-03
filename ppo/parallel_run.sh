python ppo.py -n cnn -s small -c False -g 0.99 &
python ppo.py -n cnn -s small -c True -g 0.99 &
python ppo.py -n cnn -s wide -c False -g 0.99 &
python ppo.py -n cnn -s wide -c True -g 0.99 &
python ppo.py -n cnn -s small -c False -g 0.95 &
python ppo.py -n cnn -s small -c True -g 0.95 &
python ppo.py -n cnn -s wide -c False -g 0.95 &
python ppo.py -n cnn -s wide -c True -g 0.95 
