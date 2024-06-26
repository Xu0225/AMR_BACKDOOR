import os

#models = ["CNN2", "CLDNNLikeModel", "LSTMModel", "GRUModel"]  # 添加你要运行的模型名字
models = ["CNN2"]  # 添加你要运行的模型名字


#trigger_types = ["badnet", "random_location", "hanning","spectrum_shift", "phase_shift", "remapped_awgn"]  # 添加你要运行的触发器类型
trigger_types = ["badnet", "random_location","hanning","spectrum_shift", "remapped_awgn"]  # 添加你要运行的触发器类型
#trigger_types = ['badnet']
reps = ['IQ']

for model in models:
    for rep in reps:
        for trigger_type in trigger_types:
            command = f"python ./STRIP/STRIP.py --TRIGGER_TYPE {trigger_type} --POS_RATE 0.1 --EPOCH 100 --MODEL_NAME {model} --GPU_NUM 0 > ./results/{model}_{trigger_type}_STRIP.txt"
            command = f"python ./SS/SS.py --TRIGGER_TYPE {trigger_type} --POS_RATE 0.1 --EPOCH 100 --MODEL_NAME {model} --GPU_NUM 0 > ./results/{model}_{trigger_type}_SS.txt"
            command = f"python ./AC/AC.py --TRIGGER_TYPE {trigger_type} --POS_RATE 0.1 --EPOCH 100 --MODEL_NAME {model} --GPU_NUM 0 > ./results/{model}_{trigger_type}_AC.txt"
            command = f"python ./FP/fine_pruning.py --TRIGGER_TYPE {trigger_type} --POS_RATE 0.1 --EPOCH 100 --MODEL_NAME {model} --REP {rep} > ./results/{rep}_{model}_{trigger_type}_FP.txt"

            result = os.system(command)

            if result != 0:
                print(f"Command failed: {command}")