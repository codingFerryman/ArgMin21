import sys
from typing import List

from train import run, report

"""
Configs:

basic/roberta-base.json
basic/roberta-large.json
basic/bert-base.json
basic/bert-large.json
basic/albert-base.json

"""


def main(args: List[str]):
    argv = {a.split('=')[0]: a.split('=')[1] for a in args[1:]}
    config_or_modelpath = argv.get('model', None)
    assert config_or_modelpath is not None
    experiment_report = run(config_or_modelpath, cuda_device="0")
    report(experiment_report)


if __name__ == '__main__':
    main(sys.argv)
