import os
import kaggle
from zipfile import ZipFile
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_PATH = os.path.join(BASE_DIR, '../data/raw')
INTERIM_DATA_PATH = os.path.join(BASE_DIR, '../data/interim')
CONTEST_NAME = 'nlp-getting-started'
ZIP_FILE_PATH = os.path.join(RAW_DATA_PATH, CONTEST_NAME + '.zip')
FULL_TRAIN_DATA_PATH = os.path.join(RAW_DATA_PATH, 'train.csv')
TEST_DATA_PATH = os.path.join(RAW_DATA_PATH, 'test.csv')
TRAIN_DATA_PATH = os.path.join(INTERIM_DATA_PATH, 'train.csv')
VAL_DATA_PATH = os.path.join(INTERIM_DATA_PATH, 'val.csv')
TRAIN_DATA_RATIO = 0.9
RANDOM_STATE = 42

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

def load_full_raw_train_df() -> pd.DataFrame:
    """
    Loads the full raw training data
    If the data wasn't downloaded, it downloads it
    :return: Pandas DataFrame containing the raw train data
    """
    if not os.path.exists(FULL_TRAIN_DATA_PATH):
        download_dataset()
    return pd.read_csv(FULL_TRAIN_DATA_PATH)

def load_raw_test_df() -> pd.DataFrame:
    """
    Loads the raw testing data
    If the data wasn't downloaded, it downloads it
    :return: Pandas DataFrame containing the raw train data
    """
    if not os.path.exists(TEST_DATA_PATH):
        download_dataset()
    return pd.read_csv(TEST_DATA_PATH)

def create_train_val_split() -> None:
    """
    Splits the raw train data from the contest into 90% train data and 10% val data .
    """
    df = load_full_raw_train_df()
    train_no_samples = round(TRAIN_DATA_RATIO * df.shape[0])
    train = df.sample(train_no_samples, random_state=RANDOM_STATE)
    val = df.drop(train.index)
    train.to_csv(TRAIN_DATA_PATH)
    val.to_csv(VAL_DATA_PATH)

def load_raw_train_df() -> pd.DataFrame:
    """
    Loads the subset of the competition train data that is assigned for training.
    :return: Pandas DataFrame containing train data
    """
    if not os.path.exists(TRAIN_DATA_PATH):
        create_train_val_split()
    return pd.read_csv(TRAIN_DATA_PATH)

def load_raw_val_df() -> pd.DataFrame:
    """
    Loads the subset of the competition train data that is assigned for validation.
    :return: Pandas DataFrame containing validation data
    """
    if not os.path.exists(VAL_DATA_PATH):
        create_train_val_split()
    return pd.read_csv(VAL_DATA_PATH)