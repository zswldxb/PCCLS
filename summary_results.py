import os
from collections import defaultdict
import re
import numpy as np


output_path = './output'

model_paths = os.listdir(output_path)

mode_dict = defaultdict(list)
for model_path in model_paths:
    all_exp_path = os.listdir(f'{output_path}/{model_path}')
    for exp_path in all_exp_path:
        mode = '_'.join(exp_path.split('_')[:2])
        mode_dict[mode].append(f'{output_path}/{model_path}/{exp_path}')


for k, v in mode_dict.items():
    all_oa, all_macc = [], []
    for f in v:
        with open(f'{f}/train.log', 'r') as file:
            content = file.readlines()[-1]
            # 正则表达式提取 best oa 和 best macc
            pattern = r"best oa: ([\d.]+), best macc: ([\d.]+)"
            match = re.search(pattern, content)
            
            
            all_oa.append(float(match.group(1)))
            all_macc.append(float(match.group(2)))
    
    all_oa = np.array(all_oa) * 100
    all_macc = np.array(all_macc) * 100
    print(f"| Model | Avg Best OA (\%) | Max Best OA (\%) | Avg Best mAcc (\%) | Max Best mAcc (\%) |")
    print(f"| {k} "
          f"| ${all_oa.mean():.3f} \pm {all_oa.std():.3f}$ | ${all_oa.max():.3f}$ "
          f"| ${all_macc.mean():.3f} \pm {all_macc.std():.3f}$ | ${all_macc.max():.3f}$ |")