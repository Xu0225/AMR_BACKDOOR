
# -*- coding:utf-8 -*-

import os

gpu_num = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num

models = ['ResNet50','CNN','VGG16']

trigger_types = ["benign","badnet","random_location","hanning","spectrum_shift","remapped_awgn"]

for model in models:
    for trigger_type in trigger_types:

        command = f"python ./img_bd_attack.py --TRIGGER_TYPE {trigger_type} --POS_RATE 0.1 --EPOCH 100 --MODEL_NAME {model}"

        # ָ������ļ�·��
        output_file = f"./log/{model}_{trigger_type}_bd_attack.txt"

        command += f" > {output_file}"

        result = os.system(command)

        if result != 0:
            print(f"Command failed: {command}")

print(f"Output saved to: {output_file}")