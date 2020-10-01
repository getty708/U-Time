import os

import pytest


from utime.hyperparameters import YAMLHParams
from utime.logging import ScreenLogger
from utime.utils.scriptutils.train import get_train_and_val_datasets 


from utime.dataset.pytorch.sleep_study_dataset import (SleepStudyDatasetPytorch,
                                                       BatchSequence)
from utime.dataset.pytorch.utils import BaseSequence


ARGS_JUST_ONE = True

BATCH_SIZE      = 16
DATA_PER_PERIOD = 10
N_CLASSES       = 5
N_CHANNELS      = 1 
logger = ScreenLogger()


@pytest.fixture(scope='module')
def hparams():
    hparams_path = os.path.join("./data", "hparams.yaml")
    hparams = YAMLHParams(hparams_path, no_log=True, no_version_control=True)
    return hparams

@pytest.fixture(scope='module')
def datasets(hparams):
    datasets, _ = get_train_and_val_datasets(hparams, False, logger)
    
    # Load data in all datasets
    for data in datasets:
        for d in data:
            d.load(1 if ARGS_JUST_ONE else None)
            d.pairs = d.loaded_pairs   # remove the other pairs
    return datasets

@pytest.fixture(scope='module')
def bseq(dataset):
    ssd = datasets[0][0]
    seq = BatchSequence(identifier=ssd.identifier,
                        sleep_study_pairs=ssd.pairs,
                        batch_size=BATCH_SIZE,
                        data_per_period=DATA_PER_PERIOD,
                        n_classes=N_CLASSES,
                        n_channels=N_CHANNELS)
    return seq




def test_datasets_01(datasets):
    """ Check Loaded dataset """
    print("dataset:")
    print(datasets)    
    # train_seqs = [d[0] for d in datasets]
    # print(dir(train_seqs[0]))

    
def test_BatchSequence__get_period__01(datasets, bseq):
    """ Check Loaded dataset """    
    start = 1440
    end   = 800
    ss = ssd.pairs[0]
    (x, t) = seq.get_period(ss.identifier, start)
    print(x[0].shape)
    print(t)


def test_SleepStudyDatasetPytorch__init__01(datasets):
    train_seqs = [d[0] for d in datasets]
    print(train_seqs)
    ds = SleepStudyDatasetPytorch(train_seqs)
    

    
    
    
