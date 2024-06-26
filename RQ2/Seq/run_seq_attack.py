
# -*- coding:utf-8 -*-

import os

gpu_num = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num


models = ["CNN2", "CLDNNLikeModel","GRUModel"]

trigger_types = ["benign","badnet", "random_location", "hanning","spectrum_shift", "remapped_awgn","phase_shift"]
# trigger_types = ["benign"]

reps = ["AP","FFT","IQ"]

for rep in reps:
  for model in models:
      for trigger_type in trigger_types:

          command = f"python ./main.py --TRIGGER_TYPE {trigger_type} --POS_RATE 0.1 --EPOCH 100 --MODEL_NAME {model} --REP {rep}"
          output_file = f"./log/{rep}_{model}_{trigger_type}_bd_attack_epoch_100_pos_0.1.txt"
          command += f" > {output_file}"
          result = os.system(command)

          if result != 0:
              print(f"Command failed: {command}")