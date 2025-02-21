# Disaster Prediction
#### Predict which Tweets are about real disasters and which ones are not

## About the Project

This project is for the Kaggle competition
[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data).
The goal of this project is to build a machine learning model that 
predicts which Tweets correspond to real disasters and which are not.

This repo contains the notebooks used by the author to experiment
with different models and techniques. The final output is in the
disaster_prediction directory. It contains code that is used inside 
the notebooks and contains the code to train the model and use it for
inference. Moreover, there is a script that you can use to quickly
train the model, evaluate it, and make predictions.

## Getting Started

### Prerequisites

To run the code in this repo you have to install conda. You can
download it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Installation

This repo contains a `Makefile` that you can use to quickly
create the conda environment and install the required packages.

To work locally you should follow these steps:

1. Clone the repo
```sh
git clone https://github.com/ibrahimhabibeg/disaster_prediction.git
```
or if you are using GitHub CLI
```sh
gh repo clone ibrahimhabibeg/disaster_prediction
```

2. Change the directory to the repo
```sh
cd disaster_prediction
```

3. Create the conda environment
```sh
make create_environment
```

4. Activate the conda environment
```sh
conda activate disaster-prediction
```
## Usage

To train the model, evaluate it, and make predictions you can use the
`script.py` script. The script documentation is in the `docs/script.md`
file. To run the script you can use the following command:

```sh
python script.py --help
```

## Project Structure

The project is structured as follows:

```
disaster_prediction/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── predictions/
│   └── submissions/
│
├── disaster_prediction/
│   ├── dataset.py            # Downloads, splits, and loads the dataset
│   ├── model_controller.py   # Trains, evaluates, and makes predictions
│   ├── model_specs.py        # Defines a class that contains the model specs
│   ├── utils.py              # Contains utility functions
│   ├── models/              
│   │   ├── __init__.py
│   │   ├── small.py          # Contains the small model
│   │   └── large.py          # Contains the large model
│   └── __init__.py
│
├── notebooks/               # Contains the notebooks used for experimentation
│
├── script.py                # Script to train, evaluate, and make predictions
│
├── Makefile                 # Contains commands to create the conda environment
│
├── environment.yml          # Contains the conda environment
│
├── docs/                    # Contains documentation for the script.py
│
├── license
│
└── README.md
```

## About the Models

Three are two models in the `models` directory. The `small.py` model
and the `large.py` model.

The `small.py` model is based on the bert-base model with a linear 
layer on top of it. It is fine-tuned on the dataset. This model 
ignores the `keyword` column.

The `large.py` embeds the `text` column using the bert-large model
and the `keyword` column using the bert-base model. The output of
these two models is concatenated and passed through a simple
classification head made of two linear layers with a ReLU activation
function.