import os

#trigger_types = ["random_location","spectrum_shift"]  # 添加你要运行的触发器类型
trigger_types = ['benign',"badnet","hanning","random_location","spectrum_shift"]
#trigger_types = ['benign']
for trigger_type in trigger_types:
    #command = f"python ./main.py --TRIGGER_TYPE {trigger_type} "
    command = f"python ./main_stacking.py --TRIGGER_TYPE {trigger_type} "
    # 指定输出文件路径
    output_file = f"./log/{trigger_type}_multi_mode_hybrid.txt"

    # 通过在命令中添加重定向输出到文件的部分，指定文件编码（在这里使用UTF-8）
    command += f" > {output_file}"

    result = os.system(command)

    if result != 0:
        print(f"Command failed: {command}")