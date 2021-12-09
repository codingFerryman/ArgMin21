from typing import Union

from datetime import datetime

from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm, trange
from datasets import load_metric

from dataset_transformers import TransformersSentencePairDataset
from classifier_transformers import TransformersSentencePairClassifier
from utils import set_seed, get_logger, get_project_path

LOG_LEVEL = "INFO"


def training(config_path: Union[str, Path], model_class=TransformersSentencePairClassifier):
    # Set logger
    logger = get_logger("training", level=LOG_LEVEL)

    # Load config and model
    assert Path(config_path).is_file(), f"This config file doesn't exist: {config_path}"
    model = model_class(config_path=config_path)
    experiment_config = model.get_experiment_config()
    tokenizer_config = model.get_tokenizer_config()
    trainer_config = model.get_trainer_config()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    logger.debug("Model loaded")

    # Set seed
    set_seed(experiment_config.get('seed', 2021))

    # Saving configurations
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = experiment_config.get("name",  "default")
    output_path = Path(get_project_path(), "models", f"{config_name}_{now}_state.pt")

    if not output_path.exists():
        output_path.mkdir(parents=True)
    experiment_config["output_dir"] = output_path.as_posix()
    logger.info(f"Output path: {experiment_config['output_dir']}")

    # Load data
    if tokenizer_config['type'] == "transformer":
        dataset_class = TransformersSentencePairDataset
    else:
        raise NotImplementedError(f"{tokenizer_config['type']} is not supported ...")
    train_dataset = dataset_class(tokenizer_config, "train")
    dev_dataset = dataset_class(tokenizer_config, "dev")
    batch_size = trainer_config.get('batch_size', 128)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=batch_size)
    logger.debug("Data loaded")

    # Optimizer
    learning_rate = trainer_config.get('learning_rate', 5e-5)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Scheduler
    num_epochs = trainer_config.get('epoch', 5e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(train_dataloader)
    )

    # Epoch
    epoch_no_improve = 0
    min_val_loss = 100
    for epoch in range(num_epochs):
        # Start training
        model.train()
        train_loss = 0.0
        # Batch
        with tqdm(train_dataloader, unit=' batch') as tepoch:
            for idx, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch+1} Training  ")

                # get the inputs
                batch = {k: v.to(device) for k, v in batch.items()}

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass to get outputs
                outputs = model(**batch)

                # Calculate Loss
                loss = outputs.loss
                loss.backward()
                train_loss += loss.item()

                # Updating parameters
                optimizer.step()

                tepoch.set_postfix(loss=train_loss/(idx+1), lr=lr_scheduler.get_last_lr())
        # train_loss_epoch = train_loss / len(train_dataloader)

        # Start validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            with tqdm(eval_dataloader, unit=' batch') as tepoch:
                for idx, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch+1} Validation")
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()
                    tepoch.set_postfix(val_loss=val_loss/(idx+1))
        val_loss_epoch = val_loss / len(eval_dataloader)

        # Early Stopping
        if val_loss_epoch <= min_val_loss:
            min_val_loss = val_loss_epoch
            epoch_no_improve = 0
        else:
            epoch_no_improve += 1
            if epoch_no_improve >= trainer_config.get("early_stopping_patience", 1):
                logger.info("Early Stopped!")
                torch.save(model.state_dict(), Path(output_path, "model_state.pt"))
                return model, epoch+1, output_path

        # Decay Learning Rate
        lr_scheduler.step()
    logger.info("Training Finished!")
    torch.save(model.state_dict(), Path(output_path))
    return model, num_epochs, output_path

