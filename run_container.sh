docker run --gpus=all -it --rm --name=collossl --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 -p 6015:6015 -v /mnt/nfs/projects/usense/:/mnt -v $(pwd):/workspace/ tensorflow/tensorflow:latest-gpu

