import os
import pandas as pd


def load_data(file_name: str) -> pd.DataFrame:
    try:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_directory, "..", "dataset", file_name)
        return pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_name}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the dataset: {str(e)}")
        return None