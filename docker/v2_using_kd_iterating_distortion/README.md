# Docker instructions
1. Build the docker:
```shell
docker build -t knapsack_secure_inference:v2_using_kd_iterating_distortion -f docker/v2_using_kd_iterating_distortion/Dockerfile .
docker tag knapsack_secure_inference:v2_using_kd_iterating_distortion ajevnisek/knapsack_secure_inference:v2_using_kd_iterating_distortion
docker push ajevnisek/knapsack_secure_inference:v2_using_kd_iterating_distortion
```
2. Then run the docker:
On the A5000:
```shell
#docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/exports/snl_with_betas/cifar100-new-split/snl_with_betas_15000_local.sh -e RUN_SCRIPT=/local_code/scripts/snl_with_alphas_and_betas_generic.sh  -it snl-amir:v2
```
On runai:
```shell
runai submit --name amir-knapsack-iterate-distortions -g 1.0 -i ajevnisek/knapsack_secure_inference:v2_using_kd_iterating_distortion -e WORK_DIR=/storage/jevnisek/knapsack/ --pvc=storage:/storage
```
