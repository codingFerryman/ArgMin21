cuda_device=0
report_path=/root/autodl-nas/argmin21/report.csv
submit_dir=/root/autodl-nas/argmin21/submissions/

export TRANSFORMERS_VERBOSITY=error
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

export TRANSFORMERS_CACHE=/root/autodl-tmp/transformers_cache/

export MODEL_LOGGING_PATH=/root/autodl-tmp/argmin21/logging/

for model_config in albert-base.json bert-base.json roberta-base.json; do
  python \
    /root/argmin21/src/main.py \
    model=basic/$model_config \
    cuda=$cuda_device \
    report=$report_path \
    submit=$submit_dir
done
