docker run --name 'kevin_dec2' -p 6006:6006 --memory 14g --shm-size 10g -it -d --gpus all --runtime=nvidia -v /home/kevinmf94/shared:/home/kevinmf94/shared pytorch:pytorch /bin/bash