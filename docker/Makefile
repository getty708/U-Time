# Const - Image
REPO=yoshimura/utime
REPO_REMOTE=192.168.7.151:5000/yoshimura/utime
TAG=pytorch-1.6.0-cuda10.1-cudnn7-runtime

# Const - Container
GPU=0
PORT_NOTE="9500:8888"
CONTAINER_NAME="utime"


build: Dockerfile-gpu
	docker build -t $(REPO):$(TAG) -f Dockerfile-gpu .
	docker inspect $(REPO):$(TAG) > .$@

.PHONY: run
run: 
	docker run --gpus all \
	   --name $(CONTAINER_NAME) \
	   -e NVIDIA_VISIBLE_DEVICES=$(GPU) \
	   -e CUDA_VISIBLE_DEVICES=$(GPU) \
	   -v $(DIR_CODE):/home/bob/U-Time \
	   --shm-size=8gb \
	   --privileged \
	   -p $(PORT_NOTE) \
	   -it $(REPO):$(TAG) jupyter notebook --allow-root --ip 0.0.0.0


.PHONY: push
push: build
	docker image tag $(REPO):$(TAG) $(REPO_REMOTE):$(TAG)
	docker push $(REPO_REMOTE):$(TAG)
