
# -*- coding:utf-8 -*-

import os
import time
import shutil

st = time.time()
trigger_types = ["benign", "badnet", "random_location", "hanning","spectrum_shift", "remapped_awgn","phase_shift"]  # �����Ҫ���еĴ���������
#reps = ["AP","FFT","IQ"]
#trigger_types = ["phase_shift"]  # �����Ҫ���еĴ���������
reps = ["IQ"]


def delete_all_files_in_folder(folder_path):
    try:
        # ֱ�ӵݹ�ɾ���ļ��м�������
        shutil.rmtree(folder_path)
        print(f"'{folder_path}'is deleted")

    except Exception as e:
        print(f"error in deleting folder: {e}")

# ���� models �� trigger_types
for rep in reps:
    for trigger_type in trigger_types:
        # ���������ַ���
        command = f"python ./generate_img.py --TRIGGER_TYPE {trigger_type} --POS_RATE 0.1 --REP {rep}"

        # ִ������
        result = os.system(command)
        

        # �������ִ���Ƿ�ɹ�
        if result != 0:
            print(f"Command failed: {command}")
            
        
        # ʹ��ʾ��
        folder_to_delete = f"/root/zx/Thesis_Code/datasets/constellation_{trigger_type}/X_train_badnet"
        delete_all_files_in_folder(folder_to_delete)
        folder_to_delete = f"/root/zx/Thesis_Code/datasets/constellation_{trigger_type}/X_test_badnet"
        delete_all_files_in_folder(folder_to_delete)

et = time.time()
print('finished in ',et-st, 's')
