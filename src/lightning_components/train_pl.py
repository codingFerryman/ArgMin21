# sys.path.extend(['~/Documents/Workspace/_Activated/argmin21'])
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.lightning_components.classifier_pl import LitProgressBar, AVAIL_GPUS, KPMClassifier
from src.lightning_components.dataset_pl import KPMDataModule
from src.utils import set_seed

set_seed(42)

bar = LitProgressBar()

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min'
)

trainer = Trainer(
    max_epochs=30,
    precision=16,
    gpus=AVAIL_GPUS,
    auto_lr_find=True,
    callbacks=[
        bar,
        EarlyStopping(monitor="val_loss", patience=5)
    ],
    num_sanity_val_steps=0,
    # accelerator="cpu",
    # limit_train_batches=2.0,
    # limit_val_batches=1.0,
    # limit_test_batches=10.0
)

check_model_name = 'roberta-base'

dm = KPMDataModule(check_model_name)
dm.setup("fit")

dm_model = KPMClassifier(
    check_model_name,
    customized_layers=False,
    num_labels=2,
    task_name='kp_topic_pair',
    learning_rate=3e-5,
    loss_fct=None,
    pos_weight=1.
)
trainer.fit(dm_model, dm)
