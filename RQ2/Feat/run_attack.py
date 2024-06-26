import os

models = ['CART','XGBoost', 'LightGBM']
trigger_types = ["benign","badnet","random_location", "hanning","spectrum_shift", "phase_shift", "remapped_awgn"]  # 添加你要运行的触发器类型
#trigger_types = ["benign"]
views = ['basic','time','expert']
#views = ['time','expert']
for view in views:
    for model in models:
        for trigger_type in trigger_types:
            command = f"python ./main.py --TRIGGER_TYPE {trigger_type} --VIEW {view} --MODEL_NAME {model}"
            # 指定输出文件路径
            output_file = f"./log/{view}_{model}_{trigger_type}_bd_attack.txt"

            # 通过在命令中添加重定向输出到文件的部分，指定文件编码（在这里使用UTF-8）
            command += f" > {output_file}"

            # 执行命令
            result = os.system(command)

            if result != 0:
                print(f"Command failed: {command}")