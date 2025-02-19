import os.path
from typing import Annotated
import typer
from rich.console import Console
from rich.table import Table
from disaster_prediction.dataset import download_dataset, load_raw_train_df, load_raw_val_df, load_raw_test_df
from disaster_prediction.model_specs import ModelSpecs
from disaster_prediction.models.large import model_specs as large_model_specs
from disaster_prediction.models.small import model_specs as small_model_specs
from disaster_prediction.model_controller import ModelController
import pandas as pd

app = typer.Typer()

console = Console()
error_console = Console(stderr=True)

model_specs_map = {
    'large': large_model_specs,
    'small': small_model_specs
}

def validate_model(model: str):
    if model not in model_specs_map.keys():
        error_console.print('Invalid model. Options are: large, small')
        raise typer.Exit(code=1)

def validate_model_path_or_get_default(model_specs: ModelSpecs, model_path: str):
    if model_path is None or not os.path.exists(model_path):
        return f'./models/{model_specs.model_name}.pt'
    return model_path

def get_df_from_data_path_or_desc(data:str) -> pd.DataFrame:
    if data == 'train':
        return load_raw_train_df()
    elif data == 'val':
        return load_raw_val_df()
    elif data == 'test':
        return load_raw_test_df()
    else:
        return pd.read_csv(data)

@app.command()
def download(
        force: Annotated[bool, typer.Option('--force/--no-force', '-f/-F',
                                            help='force the download if the file already exists')] = False,
        quiet: Annotated[bool, typer.Option('--quiet/--no-quiet', '-q/-Q', help='suppress verbose output')] = False):
    download_dataset(force, quiet)

@app.command()
def train(model: Annotated[str, typer.Argument(help='Model to train. Options: large, small')],
          model_path: Annotated[str, typer.Option(help='Path to save the model')] = None):
    validate_model(model)
    model_specs = model_specs_map[model]
    model_path = validate_model_path_or_get_default(model_specs, model_path)
    model_controller = ModelController(model_specs, model_path)
    train_df = load_raw_train_df()
    model_controller.train(train_df)
    typer.echo(f'Model trained and saved to {model_path}')


@app.command()
def predict(model: Annotated[str, typer.Argument(help='Model to predict with. Options: large, small')],
            data: Annotated[str, typer.Argument(
                help='Path to the data to predict on. '
                     'The data should be a CSV file with id, text, and keyword (can be null) columns. '
                     'Other valid values are "train", "val", and "test".')],
            output: Annotated[str, typer.Argument(help='Path to save the predictions')] = None,
            force_train: Annotated[bool, typer.Option('--force-train/--no-force-train', '-f/-F',
                                                      help='Force training the model before predicting')] = False,
            model_path: Annotated[str, typer.Option(help='The path where the model is saved')] = None):
    model_specs = model_specs_map[model]
    model_path = validate_model_path_or_get_default(model_specs, model_path)
    model_controller = ModelController(model_specs, model_path)
    if force_train or not model_controller.is_trained:
        train_df = load_raw_train_df()
        model_controller.train(train_df)
    data = get_df_from_data_path_or_desc(data)
    predictions = model_controller.predict(data)
    if output is None:
        output = f'./data/predictions/{model_specs.model_name}.csv'
    data['target'] = predictions
    data = data[['id', 'target']]
    data.to_csv(output, index=False)
    typer.echo(f'Predictions saved to {output}')

@app.command()
def evaluate(model: Annotated[str, typer.Argument(help='Model to evaluate. Options: large, small')],
             data: Annotated[str, typer.Argument(
                 help='Path to the data to evaluate on. '
                      'The data should be a CSV file with both target and text columns and an additional keyword column '
                      'if using the large model. Other valid values are "train" and "val"')] =
                'val',
             force_train: Annotated[bool, typer.Option('--force-train/--no-force-train', '-f/-F',
                                                       help='Force training the model before predicting')] = False,
             model_path: Annotated[str, typer.Option(help='The path where the model is saved. Uses '
                                                          './models/{model_specs.model_name}.pt if no value is passed.')] = None):
    model_specs = model_specs_map[model]
    model_path = validate_model_path_or_get_default(model_specs, model_path)
    model_controller = ModelController(model_specs, model_path)
    if force_train or not model_controller.is_trained:
        train_df = load_raw_train_df()
        model_controller.train(train_df)
    if data == 'test':
        error_console.print("Can't predict on the test dataset.")
        typer.Exit(code=1)
    data = get_df_from_data_path_or_desc(data)
    metrics = model_controller.evaluate(data)
    table = Table(title='Evaluation Metrics')
    table.add_column('Metric')
    table.add_column('Value')
    for metric, value in metrics.items():
        table.add_row(metric, str(value))
    console.print(table)

if __name__ == '__main__':
    app()