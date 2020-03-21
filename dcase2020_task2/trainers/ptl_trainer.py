from typing import NoReturn
from trainers import BaseTrainer
from experiments import BaseExperiment
import pytorch_lightning as pl


class PTLTrainer(BaseTrainer):

    def __init__(self, **config):
        self.config = config
        self.trainer = pl.Trainer(**config)

    def fit(self, experiment: BaseExperiment) -> NoReturn:
        self.trainer.fit(model=experiment)

    def test(self, experiment: BaseExperiment) -> NoReturn:
        self.trainer.test(model=experiment)
