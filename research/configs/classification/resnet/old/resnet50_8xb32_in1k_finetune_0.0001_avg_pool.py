_base_ = [
    '../_base_/models/resnet50_avg_pool.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256_finetune_0.0001.py', '../_base_/default_runtime.py'
]

relu_spec_file = None
load_from = "./mmlab_models/classification/resnet50_8xb32_in1k_20210831-ea4938fc.pth"
work_dir = "./outputs/classification/resnet50_8xb32_in1k/finetune_0.0001_avg_pool"












# "/storage/yakir/secure_inference/research/configs/classification/resnet/benchmark/3x4.py",
# "/storage/yakir/secure_inference/research/configs/classification/resnet/benchmark/3x3.py",
# "/storage/yakir/secure_inference/research/configs/classification/resnet/benchmark/4x4.py",
# "/storage/yakir/secure_inference/research/configs/classification/resnet/benchmark/2x3.py",

# "/storage/yakir/secure_inference/research/configs/classification/resnet/benchmark/0.08.py"
# "/storage/yakir/secure_inference/research/configs/classification/resnet/benchmark/0.05.py"
# "/storage/yakir/secure_inference/research/configs/classification/resnet/benchmark/0.1.py"
# "/storage/yakir/secure_inference/research/configs/classification/resnet/benchmark/0.15.py"

# "/storage/yakir/secure_inference/research/configs/classification/resnet/benchmark/3x4_naive.py"
# "/storage/yakir/secure_inference/research/configs/classification/resnet/benchmark/baseline.py"
# "/storage/yakir/secure_inference/research/configs/classification/resnet/benchmark/3x3_naive.py"
# "/storage/yakir/secure_inference/research/configs/classification/resnet/benchmark/4x4_naive.py"
# "/storage/yakir/secure_inference/research/configs/classification/resnet/benchmark/2x3_naive.py"
#
# import pickle
# content = pickle.load(open("0.05.pickle", 'rb'))
# for k in content:
#     content[k][:, 0] = 2
#     content[k][:, 1] = 3
#
# pickle.dump(file=open("2x3_naive.pickle", 'wb'), obj=content)

