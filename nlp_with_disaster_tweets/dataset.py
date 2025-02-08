import os
import kaggle
from zipfile import ZipFile
import pandas as pd

RAW_DATA_PATH = '../data/raw'
CONTEST_NAME = 'nlp-getting-started'
ZIP_FILE_PATH = os.path.join(RAW_DATA_PATH, CONTEST_NAME + '.zip')
TRAIN_DATA_PATH = os.path.join(RAW_DATA_PATH, 'train.csv')
TEST_DATA_PATH = os.path.join(RAW_DATA_PATH, 'test.csv')

def download_dataset(force:bool = True, quiet: bool = True) -> None:
    """
    Downloads the contest data from Kaggle
    :param force: force the download if the file already exists (default True)
    :param quiet: suppress verbose output (default is True)
    :return:
    """
    kaggle.api.competition_download_files(CONTEST_NAME, path=RAW_DATA_PATH, force=force, quiet=quiet)
    with ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_PATH)
    os.remove(ZIP_FILE_PATH)

def load_raw_train_df() -> pd.DataFrame:
    """
    Loads the raw training data
    If the data wasn't downloaded, it downloads it
    :return: Pandas DataFrame containing the raw train data
    """
    if not os.path.exists(TRAIN_DATA_PATH):
        download_dataset()
    return pd.read_csv(TRAIN_DATA_PATH)

def load_raw_test_df() -> pd.DataFrame:
    """
    Loads the raw testing data
    If the data wasn't downloaded, it downloads it
    :return: Pandas DataFrame containing the raw train data
    """
    if not os.path.exists(TEST_DATA_PATH):
        download_dataset()
    return pd.read_csv(TEST_DATA_PATH)