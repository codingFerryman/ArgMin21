# Key Point Matching via Transformers

This repository works on the shard task Key Point Analysis (KPA) shared task ([link](https://github.com/IBM/KPA_2021_shared_task)) which is used in the 8th Workshop on Argument Mining co-located with EMNLP 2021 ([link](https://2021.argmining.org/)). This project is only for self-study use, especially for the effectiveness of K-Fold cross validation, class weights in a loss function, and adding more features. I did not participate this workshop.

## Setup
Creating a virtual environment, then execute:
```bash
cd PATH-TO-REPO-DIR
bash ./setup_env.sh
```
Note: if you are running on a machine without CUDA device, please remove *+cu113* in [requirements.txt](./requirements.txt) to install the CPU version of PyTorch. You can also manually install the dependencies.

## Repository Structure
### Configurations
The training configurations are stored in [config](./config/) and are in JSON format:
```JSON
{
  "name": "Name of task",
  "seed": (int) random-seed,
  "model_name": "Model name in Huggingface Transformers",
  "data_config": {
    "load_ratio": (float) ratio for partially data loading,
    "add_info": null or additional information separated by comma(s) (supported information: topic,stance)
  },
  "model_config": {
  },
  "tokenizer_config": {
    "max_length": (int) the maximum length of tokenizer (sequence shorter than this value will be padded and longer than this value will be truncated), 
    "return_tensors": "pt" (do NOT change. Please be aware that this project doesn't support TensorFlow at least now)
  },
  "trainer_config": {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 64,
    "eval_accumulation_steps": 30,
    "learning_rate": 3e-5,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.00001,
    "metric_for_best_model": "eval_loss",
    "fp16": true
  },
  "eval_config": {
    "mode": "plain",
    "batch_size": (int) batch size in evaluation and prediction procedure
  }
}
```
Notes:
- Please refer to [Huggingface Models Hub](https://huggingface.co/models?language=en&library=pytorch&sort=downloads) for the name of models.
- Please refer to [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) for the parameters in *trainer_config*.
- *mode* in *eval_config* does not have any effect but please keep it there for compatible reason.

### Source Code
Most of the source code are in [src](./src/). 
- [main.py](./src/main.py) can execute the experiment from a given configuration. You can also execute the training from a bash script refer to [execute.sh](./execute.sh)
- [train.py](./src/train.py) and [train_kfold.py](./src/train_kfold.py) load pre-trained models from Huggingface Models and execute fine-tuning based on given configurations. The second file execute training with K-Fold cross validation, and the folds will be splitted based on the *topics* information. 
- [dataset.py](./src/dataset.py): PyTorch Dataset wrapper for dataset loading
- [classifier.py](./src/classifier.py): a wrapper of the Trainer implementation in Huggingface Transformers.
- [kfolds.py](./src/kfolds.py) is for K-Fold cross validation, which can load, split, and return specific fold.
- [evaluate.py](./src/classifier.py): functions for performance evaluation and creating the submission file.
- [predict.py](./src/predict.py) is used for loading a trained model and predict the labels of given data.
- [utlis.py](./src/utils.py) miscelluous functions.



