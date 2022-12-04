import subprocess
import os
import time


class JobFetcher:
    def __init__(self):
        pass

    def job(self):

        ratio = 0.1
        iter_to_checkpoint = ["/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/snr/iter_20000.pth",
                              "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/snr/iter_20000.pth"]
        config = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline.py"
        dataset = "ade_20k_256x256"
        base_output_path = f"/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/snr_2"
        output_path = os.path.join(base_output_path, "channel_distortions")
        block_size_spec_file_name = "/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/snr_2/block_size_spec.pickle"
        script_path = "/home/yakir/PycharmProjects/secure_inference/research/distortion/channel_distortion.py"
        knapsack_script_path = "/home/yakir/PycharmProjects/secure_inference/research/knapsack/multiple_choice_knapsack.py"
        params_name = "MobileNetV2_256_Params_2_Groups"

        for iteration in range(0, 1):
            for batch_index in [0, 1, 2]:
                jobs = []
                for gpu_id in [0, 1]:
                    job_params = [
                        "--batch_index", f"{2 * batch_index + gpu_id}",
                        "--gpu_id", f"{gpu_id}",
                        "--dataset", dataset,
                        "--config", config,
                        "--checkpoint", iter_to_checkpoint[iteration],
                        "--iter", f"{iteration}",
                        "--block_size_spec_file_name", block_size_spec_file_name,
                        "--output_path", output_path,
                        "--params_name", params_name
                    ]
                    jobs.append(["python", script_path] + job_params)

                running_jobs = [subprocess.Popen(job) for job in jobs]

                for running_job in running_jobs:
                    running_job.wait()

            # knapsack_job_params = [
            #
            #     "--iter", f"{iteration}",
            #     # "--block_size_spec_file_name", block_size_spec_file_name,
            #     "--output_path", output_path,
            #     '--ratio', f"{ratio}",
            #     "--params_name", params_name,
            #     "--cost_type", "Bandwidth",
            #     "--division", "512"
            #
            # ]
            # job = ["python", knapsack_script_path] + knapsack_job_params
            # running_job = subprocess.Popen(job)
            # running_job.wait()


if __name__ == "__main__":
    JobFetcher().job()