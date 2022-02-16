"""
shape_dnn

Author: Rahim Mammadli
Date:   02.12.17
Source:https://gist.github.com/Rahim16/2591a3c837456afb05e353dd8e880800#file-measure_energy-py
"""

from multiprocessing import Process, Pipe
from pynvml import *
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchnet as tnt
import torch.backends.cudnn as cudnn
from nested_dict import nested_dict
from collections import OrderedDict
import datetime
import time
from datetime import timedelta
import shape as sh
from ast import literal_eval
import pickle

cudnn.benchmark = True

DEFAULT_RECORDING_INTERVAL = 4
DEFAULT_TIMEOUT = 40
DEFAULT_SENSITIVE_WINDOW = 5
DEFAULT_SENSITIVE_THRESHOLD = 7000
DEFAULT_INPUT_DIM = "(32,32,3)"

parser = argparse.ArgumentParser(description='Energy Benchmarking for CUDA')
parser.add_argument('--input_dim', type=str, default=DEFAULT_INPUT_DIM)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--simulate', action='store_true')
parser.add_argument('--sw', dest='sensitive_window', default=DEFAULT_SENSITIVE_WINDOW, help='Sensitivity window in seconds.', type=int)
parser.add_argument('--sv', dest='sensitive_threshold', default=DEFAULT_SENSITIVE_THRESHOLD, help='Sensitivity threshold in milliwatts.', type=int)
parser.add_argument('-f', '--file', help='Output file to store the results.', default="../logfiles/power_measurement.json")
parser.add_argument('--gpu_id', help='Which GPU to perform tests on.', default=0, type=int)
parser.add_argument('--recording_interval', help='Values will be recorded if this amount of time(ms) has passed even though they don\'t change.', default=DEFAULT_RECORDING_INTERVAL, type=int)
parser.add_argument('--timeout', help='This is the max amount of time in seconds that the process will wait \
    for power to be stabilized. If it doesn\'t average power consumption will be computed.', default=DEFAULT_TIMEOUT, type=int)

def measure(conn, handle, sensitive_window=DEFAULT_SENSITIVE_WINDOW, recording_interval=DEFAULT_RECORDING_INTERVAL, timeout=DEFAULT_TIMEOUT):
    measurements = []

    min, max = None, None

    sensitivity_window = datetime.timedelta(seconds=sensitive_window)
    recording_interval = datetime.timedelta(milliseconds=recording_interval)
    timeout = datetime.timedelta(seconds=timeout)
    measurements_started = datetime.datetime.now()
    max_power_reached = False
    stable_power = None

    while True:

        current = {
            'value': nvmlDeviceGetPowerUsage(handle),
            'time': datetime.datetime.now()
        }

        if len(measurements) == 0:
            min = current
            max = current
            measurements.append(current)
            continue

        previous = measurements[len(measurements)-1]

        if current['value'] != previous['value'] or current['time'] - previous['time'] >= recording_interval:
            measurements.append(current)

        if len(measurements) > 1000000:
            conn.send("too_many_values")
            break

        if (current['value'] > max['value']):
            print("New max: %s" % current)
            max = current
        elif (current['value'] < min['value']):
            print("New min: %s" % current)
            min = current
        elif (current['time'] - max['time'] > sensitivity_window
        and current['time'] - min['time'] > sensitivity_window):
            if max['value'] - min['value'] < sensitive_window:
                conn.send("power_stabilized")
                stable_power = np.mean([min['value'], max['value']])
                print("Min: %d Max: %d Stable: %.2f" % (min['value'], max['value'], stable_power))
                break

            else: # resetting both, because at least one is stale
                max = min = current

        if  current['time'] - measurements_started > timeout:
            conn.send("power_not_stable")
            break


    conn.send({
        'measurements': measurements,
        'stable_power': stable_power
    })

def record_energy(conn, gpu_id, sensitive_window, recording_interval, timeout):
    nvmlInit()
    print("Number of GPUs: %d" % nvmlDeviceGetCount())
    # ordering for nvml is reversed for some reason in *some cases*!
    nvml_gpu_id = int(gpu_id)
    handle = nvmlDeviceGetHandleByIndex(nvml_gpu_id)
    device_name = nvmlDeviceGetName(handle)
    print("Testing Device: %s" % device_name)


    while conn.recv() != "done":
        print("beginning to measure")
        measure(conn, handle, sensitive_window, recording_interval, timeout)
        print("measurement complete")

    print("all measurements complete")
    conn.close()
    nvmlShutdown()

def default_serializer(obj):
    """Default JSON serializer."""
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    raise TypeError('Not sure how to serialize %s' % (obj,))

def to_microseconds(delta, verbose=False):
    timed = delta
    in_micro = timed.seconds * 1000000 + timed.microseconds
    if verbose:
        print(date)
        print(timed)
        print(in_micro)
    return in_micro

def simulate_shape(parent_conn, depth, width, batch_sizes, batches, input_dim, num_classes, model_class = None, device = None):
    device = torch.device("cuda:0") if device is None else device
    print('Constructing new ResNet..')
    resnet = model_class(int(depth), width, input_dim, num_classes, device=device)
    batch_results = dict()

    for batch_size in reversed(batch_sizes):
        print("New measurement[Depth:\t%d\tWidth:\t%.4f\tBatch Size:\t%d]" % (depth, width, batch_size))
        parent_conn.send("go")

        # dry runs
        for i in range(5):
            resnet.forward(batches[batch_size])

        runs = []
        simulation_start = datetime.datetime.now()
        while not parent_conn.poll():
            torch.cuda.synchronize()
            run_start = datetime.datetime.now()
            resnet.forward(batches[batch_size])
            torch.cuda.synchronize()
            run_finish = datetime.datetime.now()
            runs.append({'start': run_start,
                'finish': run_finish})
        simulation_end = datetime.datetime.now()

        measurement_status = parent_conn.recv()
        measurement_results = parent_conn.recv()

        print(measurement_status)

        if "power_not_stable" == measurement_status:
            energies = []
            last_n_runs = len(runs) / 2
            middle_run_start = runs[-last_n_runs]['start']
            last_run_finish = runs[-1]['finish']
            power_readings = []
            for measurement in measurement_results['measurements']:
                if measurement['time'] > last_run_finish:
                    break
                elif measurement['time'] > middle_run_start:
                    power_readings.append(measurement['value'])

            power = np.mean(power_readings) / 1000.0 # converted to watts
            fw = np.mean([to_microseconds(run['finish'] - run['start']) / 1000.0 for run in runs[-last_n_runs:]]) # converted to milliseconds
            batch_results[batch_size] = {
                'stable_power': power,
                'stable_fw': fw,
                'energy': power * fw # joules
            }
        elif "power_stabilized" == measurement_status:
            power = measurement_results['stable_power'] / 1000.0 # converted to watts
            fw = np.mean([to_microseconds(run['finish'] - run['start']) / 1000.0 for run in runs[-10:]]) # converted to milliseconds
            batch_results[batch_size] = {
                'stable_power': power,
                'stable_fw': fw,
                'energy': power * fw # joules
            }


        print(batch_results[batch_size])

    shape_data = {
        'batch_results': batch_results,
        'param_count': resnet.n_parameters,
        'depth': depth,
        'width': width,
        'num_classes': num_classes
    }
    return shape_data

def simulate(gpu_id, depths, widths, batch_sizes, input_dim, num_classes, log_dir=None, log_file=None, sensitive_window=DEFAULT_SENSITIVE_WINDOW, recording_interval=DEFAULT_RECORDING_INTERVAL, timeout=DEFAULT_TIMEOUT):
    parent_conn, child_conn = Pipe()
    recorder_process = Process(target=record_energy, args=(child_conn, gpu_id, sensitive_window, recording_interval, timeout))
    recorder_process.start()

    if log_file != None:
        lfile = log_file
    elif log_dir != None:
        lfile = os.path.join(log_dir, "power_measurement.json")
    else:
        raise Exception("Either log directory or log file should be provided.")

    batches = dict()
    for i in reversed(batch_sizes):
        batches[i] = Variable(torch.randn(i, input_dim[-1], input_dim[-2], input_dim[-3]).cuda(), requires_grad=False)

    print('x', flush=True)
    log_data = {
        'shapes': [],
        'options': {
            'gpu_id': gpu_id,
            'depths': depths,
            'widths': widths,
            'batch_sizes': batch_sizes,
            'input_dim': input_dim,
            'num_classes': num_classes
        }
    }

    for test_width in widths:
        for test_depth in depths:
            new_shape_result = simulate_shape(parent_conn, test_depth, test_width, batch_sizes, batches, input_dim, num_classes)
            log_data['shapes'].append(shape_data)


    print("Logging results to %s" % lfile)
    with open(lfile, 'w') as f:
        f.write(json.dumps(log_data, default=default_serializer) + '\n')

    parent_conn.send("done")
    recorder_process.join()
    return log_data


# extract json data to nd_array format of numpy
def extract_data(shapes):
    data = np.asarray([
        [
        shape["depth"],
        shape["width"],
        shape["param_count"],
        shape["num_classes"],
        batch_size,
        batch_value["stable_fw"],
        batch_value["stable_power"],
        batch_value["energy"]
        ] for shape in shapes for batch_size, batch_value in shape['batch_results'].items()]).astype(np.float)

    indexes = {
        'depth': 0,
        'width': 1,
        'param_count': 2,
        'num_classes': 3,
        'batch_size': 4,
        'fw': 5,
        'power': 6,
        'energy': 7
    }

    depths = np.unique(data[:,indexes['depth']])
    widths = np.unique(data[:,indexes['width']])
    batch_sizes = np.unique(data[:,indexes['batch_size']])


    return data, depths, widths, batch_sizes, indexes


def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))

    input_dim = literal_eval(opt.input_dim)
    widths = [1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    depths = [10, 16, 22, 28, 34, 40, 46, 52, 58, 64]
    batch_sizes = [1]


    simulation = None
    if opt.simulate:
        simulation = simulate(opt.gpu_id, depths, widths, batch_sizes, input_dim, opt.num_classes, log_file=opt.file, sensitive_window=opt.sensitive_window, recording_interval=opt.recording_interval, timeout=opt.timeout)


if __name__ == '__main__':
    main()
