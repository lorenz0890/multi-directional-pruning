from __future__ import print_function
from utils import Parser
from utils import ExperimentFactory
from os import listdir
from os.path import isfile, join
import timeit
import traceback

from tqdm import tqdm
from functools import partialmethod
import sys, os

def blockPrint():
    # https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    # https://stackoverflow.com/questions/37091673/silence-tqdms-output-while-running-tests-or-running-the-code-via-cron
    sys.stdout = open(os.devnull, 'w')
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def enablePrint():
    # https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    # https://stackoverflow.com/questions/37091673/silence-tqdms-output-while-running-tests-or-running-the-code-via-cron
    sys.stdout = sys.__stdout__
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)


def print_usage():
    print('Usage: python main.py --config /path/to/a/single/config.json')
    print('Usage: python main.py --batch /path/to/directory/of/multiple/configs/')

def main():
    if len(sys.argv) > 3:
        print('Too many arguments', flush=True)
        print_usage()
        sys.exit(1)
    if len(sys.argv) < 3:
        print('Missing arguments', flush=True)
        print_usage()
        sys.exit(1)
    single_mode = True
    if '--config' == sys.argv[1]:
        pass
    elif '--batch' == sys.argv[1]:
        single_mode = False
    else:
        print_usage()
        sys.exit(1)

    target_path = sys.argv[2]

    config = Parser()
    experiment_factory = ExperimentFactory()

    if single_mode:
        config.load(target_path)
        experiment = experiment_factory.get_experiment(config)
        experiment.dispatch()
    elif not single_mode:
        paths = [
            target_path
        ]

        for path in paths:
            enablePrint()
            print('BEGIN:', path, flush=True)
            blockPrint()
            fnames = [f for f in listdir(path) if isfile(join(path, f))]
            failed, succeeded = {}, {}
            fail_ctr, success_ctr, ctr = 0, 0, 0
            for fname in fnames:
                ctr += 1
                enablePrint()
                print('Experiment:', fname, flush=True)
                blockPrint()
                try:
                    start = timeit.default_timer()
                    config.load(path + fname)
                    experiment = experiment_factory.get_experiment(config)
                    experiment.dispatch()
                    stop = timeit.default_timer()
                    success_ctr += 1
                    succeeded[fname] = stop - start
                except Exception as e:
                    failed[fname] = traceback.format_exc()  # e
                    fail_ctr += 1
                enablePrint()
                print('Success (Current):', (success_ctr), '/', ctr, flush=True)
                print('Failed: (Current)', (fail_ctr), '/', ctr, flush=True)
                print('Progress:', success_ctr + fail_ctr, '/', len(fnames), flush=True)
                blockPrint()
            enablePrint()
            print('Success (Total):', (success_ctr), '/', len(fnames), flush=True)
            print('Failed (Total):', (fail_ctr), '/', len(fnames), flush=True)
            for key in failed:
                print(key, failed[key], flush=True)
            for key in succeeded:
                print(key, succeeded[key], flush=True)
            print('END:', path, flush=True)
            blockPrint()

if __name__ == "__main__":
    main()
