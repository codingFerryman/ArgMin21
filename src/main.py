import sys
from typing import List

from train import run, report
from train_kfold import run_kfold, report_kfold

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
    cuda_device = argv.get('cuda', "0")
    report_path = argv.get('report', None)

    if 'kfold' not in config_or_modelpath:
        experiment_report = run(config_or_modelpath, cuda_device=cuda_device)
        report(experiment_report, report_path=report_path)
    else:
        final_report, folds_predictions = run_kfold(config_or_modelpath, cuda_device=cuda_device)
        report_kfold(final_report, report_path, folds_predictions)


if __name__ == '__main__':
    main(sys.argv)
