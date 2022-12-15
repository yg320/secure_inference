from research.block_relu.params import ParamsFactory
import subprocess

script_path = "/home/yakir/PycharmProjects/secure_inference/research/block_relu/deformation_handler.py"
param_json_file = "/home/yakir/PycharmProjects/secure_inference/research/block_relu/distortion_handler_configs/resnet_COCO_164K_8_hierarchies.json"
# ./research/pipeline/dist_train.sh /home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/deeplabv3_r50-d8_512x512_4x4_80k_coco-stuff164k.py 2

params_0 = ["--batch_index", "0,2", "--gpu_id", "0", "--hierarchy_type", "blocks", "--operation", "extract", "--param_json_file", param_json_file]
# params_1 = ["--batch_index", "1,3", "--gpu_id", "1", "--hierarchy_type", "blocks", "--operation", "extract", "--param_json_file", param_json_file]
p0 = subprocess.Popen(["python", script_path] + params_0)
# p1 = subprocess.Popen(["python", script_path] + params_1)
#
# p0.wait()
# p1.wait()
#
#
# params_0 = ["--gpu_id", "0", "--hierarchy_type", "blocks", "--operation", "collect", "--param_json_file", param_json_file]
# p0 = subprocess.Popen(["python", script_path] + params_0)
# p0.wait()
#
#
# params = ParamsFactory()(param_json_file)
# l = list(set(map(len, params.LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS.values())))
# l = [max(l)]
# #
# for hierarchy_level in range(10, l[0]):
#
#     params_0 = ["--batch_index", "0,2", "--gpu_id", "0", "--hierarchy_type", "channels","--hierarchy_level", f"{hierarchy_level}",
#                 "--operation", "extract", "--param_json_file", param_json_file]
#     params_1 = ["--batch_index", "1,3", "--gpu_id", "1", "--hierarchy_type", "channels","--hierarchy_level", f"{hierarchy_level}",
#                 "--operation", "extract", "--param_json_file", param_json_file]
#
#     p0 = subprocess.Popen(["python", script_path] + params_0)
#     p1 = subprocess.Popen(["python", script_path] + params_1)
#
#     p0.wait()
#     p1.wait()
#
#     params_0 = ["--gpu_id", "0", "--hierarchy_type", "channels", "--operation", "collect", "--param_json_file", param_json_file, "--hierarchy_level", f"{hierarchy_level}"]
#     p0 = subprocess.Popen(["python", script_path] + params_0)
#     p0.wait()



# params_0 = ["--gpu_id", "0", "--hierarchy_type", "layers", "--hierarchy_level", "-1", "--operation", "collect", "--param_json_file", param_json_file]
# p0 = subprocess.Popen(["python", script_path] + params_0)
# p0.wait()

for hierarchy_level in range(1,2):
    params_0 = ["--batch_index", "0,2", "--gpu_id", "0", "--hierarchy_type", "layers","--hierarchy_level", f"{hierarchy_level}",
                "--operation", "extract", "--param_json_file", param_json_file]
    params_1 = ["--batch_index", "1,3", "--gpu_id", "1", "--hierarchy_type", "layers","--hierarchy_level", f"{hierarchy_level}",
                "--operation", "extract", "--param_json_file", param_json_file]

    p0 = subprocess.Popen(["python", script_path] + params_0)
    p1 = subprocess.Popen(["python", script_path] + params_1)

    p0.wait()
    p1.wait()

    params_0 = ["--gpu_id", "0", "--hierarchy_type", "layers", "--operation", "collect", "--param_json_file", param_json_file, "--hierarchy_level", f"{hierarchy_level}"]
    p0 = subprocess.Popen(["python", script_path] + params_0)
    p0.wait()


params_0 = ["--gpu_id", "0", "--hierarchy_type", "layers", "--operation", "get_reduction_spec", "--param_json_file", param_json_file, "--hierarchy_level", f"{2}"]
p0 = subprocess.Popen(["python", script_path] + params_0)
p0.wait()







#
# params_0 = ["--gpu_id", "0", "--hierarchy_level", "0", "--operation", "get_reduction_spec", "--param_json_file", param_json_file]
# p0 = subprocess.Popen(["python", script_path] + params_0)
# p0.wait()
#
#
#
#
# import numpy as np
# import glob
#
# files = glob.glob("/home/yakir/Data2/assets_v3/deformations/coco_stuff164k/ResNetV1c/V1/channels_0_in/*")
# for f in files:
#     a = np.load(f)
#     b = np.load(f.replace("V1/channels_0_in", "V0/channels_0_in"))
#     print(np.mean(np.abs(a-b)))