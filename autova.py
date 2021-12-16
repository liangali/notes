import os, sys, subprocess, time
from datetime import datetime
from pathlib import Path
import numpy as np

gpu_top_path = '/home/fresh/data/intel_gpu_top'
kill_gpu_top = 'sudo killall -9 intel_gpu_top'

# base_cmd = './benchmark_app -m models/resnet_v1.5_50_i8.xml -d GPU -b 1 -t 100 -use_device_mem -nstreams 16'
base_cmd = './benchmark_app -m models/ssd_mobilenet_v1_coco.xml -d GPU -b 1 -t 100 -use_device_mem -nstreams 16'

result_dict = {}

def run_cmd(cmd):
    print('###########: %s'%cmd)
    os.system(cmd)

def run_async(cmd):
    print('###########: %s'%cmd)
    p = subprocess.Popen(cmd, shell=True)

def run_multi_process(n):
    gputop_cmd = 'sudo %s -o multi_%d_gputop.log' % (gpu_top_path, n)
    multi_cmd = ''
    for i in range(n):
        multi_cmd += '%s > multi_%d_%d.log & ' % (base_cmd, n, i) 
    multi_cmd = multi_cmd[:-2]
    run_async(gputop_cmd)
    run_cmd(multi_cmd)
    run_cmd(kill_gpu_top)
    time.sleep(20)

def calc_multi(n):
    fps_list, freq_list = [], []
    for i in range(n):
        filename = 'multi_%d_%d.log' % (n, i)
        with open(filename, 'rt') as f:
            for line in f:
                if 'Throughput:' in line:
                    fps = float(line.split('Throughput:')[1].split('FPS')[0])
                    fps_list.append(fps)
    ccs0, ccs1, ccs2, ccs3 = [], [], [], []
    gputop_log = 'multi_%d_gputop.log'%n
    print(gputop_log)
    with open(gputop_log, 'rt') as f:
        for line in f:
            if line.split()[1].isnumeric():
                freq = int(line.split()[1])
                if freq > 200:
                    ccs0.append(float(line.split()[28]))
                    ccs1.append(float(line.split()[31]))
                    ccs2.append(float(line.split()[34]))
                    ccs3.append(float(line.split()[37]))
                    freq_list.append(freq)
    result_dict[n] = (fps_list, int(np.average(freq_list)), float(np.average(ccs0)), float(np.average(ccs1)), float(np.average(ccs2)), float(np.average(ccs3)))

def gen_report(n):
    head_line = 'process count, GT freq, ccs0 %, ccs1 %, ccs2 %, ccs3 %, '
    for i in range(1, n):
        calc_multi(i)
        head_line += 'proc-%d fps, ' % (i)
    head_line += 'Total fps \n'

    print(head_line)

    with open('perf.csv', 'wt') as f:
        lines = []
        f.writelines(head_line)
        for k, v in result_dict.items():
            fpslist = v[0]+[0]*(n-1-len(v[0]))
            fps_str = '%s, %.2f' % (', '.join([str(i) for i in fpslist]), float(np.sum(fpslist)))
            line = '%d, %d, %.01f, %.01f, %.01f, %.01f, %s' % (k, v[1], v[2], v[3], v[4], v[5], fps_str)
            print(line)
            lines.append(line)
        f.writelines('\n'.join(lines))

def save_data():
    folder_name = 'perf_%s' % datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    run_cmd('mv multi_*.log %s' % folder_name)
    run_cmd('mv perf.csv %s' % folder_name)

def execute(n):
    for i in range(1, n):
        run_multi_process(i)
    gen_report(n)
    save_data()

execute(2)

print('done')