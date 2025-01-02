
import re
import sys

import numpy as np
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict

# from dassl.utils import check_isfile, listdir_nohidden


from glob import glob
from os.path import exists, join
import shutil

def get_type_files(dirs, filetype=None, recursive=False):
    if not exists(dirs):
        return "Error: No such file or directory: {}".format(dirs)
    if filetype is None:
        return 'file_type needed!'
    if recursive:
        # 递归查找
        file_list = glob(join(dirs, '**/*'+filetype), recursive=True)
    else:
        file_list = glob(join(dirs, '*'+filetype), recursive=False)
    return file_list


if __name__ == '__main__':
    # 指定目录
    folder = sys.argv[1]
    print(folder)
    # 指定文件类型
    file_type = '.txt'
    # 获取结果，recursive表示递归查找
    result = get_type_files(folder, file_type, recursive=True)
    # print(result, end='\n')
    metric = {
        'name': 'Best Accuracy',
        'regex': re.compile(r'Best Accuracy: ([\.\deE+-]+)')
        # 'regex': re.compile(r'\* accuracy: ([\.\deE+-]+)')
    }
    acc = []
    for fpath in result:
        output = OrderedDict()
        with open(fpath, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                match = metric['regex'].search(line)
                if match:
                    if 'file' not in output:
                        output['file'] = fpath
                    num = float(match.group(1))
                    acc.append(num)
                    name = metric['name']
                    output[name] = num
        if output:
            print(fpath.split("/")[11:13], ":", num, '\n')
    print(sum(acc[:3]) / 3)
    print(sum(acc[3: 6]) / 3)
    print(sum(acc[6: 9]) / 3)
    print(sum(acc[9: 12]) / 3)
    print(sum(acc) / len(acc))



