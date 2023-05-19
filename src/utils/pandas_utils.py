
from typing import Dict, List
import pandas as pd

def flatten_dict(d, parent_key='', sep='|'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def reshape_nested_dict(
    nested_dict: Dict,
    index: List[str],
    columns: List[str],
    rows: List[str],
    csv_file: str = None,
) -> pd.DataFrame:
    """Convert a nested dictionary into Pandas DataFrame object.

    Args:
        nested_dict (dict): A nested dictionary.
        index (list of strings): A list of string names for each level of keys in
            the nested dictionary `nested_dict`.
        columns (list of strings): List of column names. The names must be in `index`
            and the resulting dataframe will have hierarchical column headers in the
            same order as in `columns`.
        rows (list of strings): List of row index names. The names must be in `index`
            and the resulting dataframe will have multi-index index names in the same
            order as in `rows`, if len(rows) > 1. Otherwise, if len(rows) == 1, the
            dataframe should contain just 1 level of row index.
    
    Return:
        df (pd.DataFrame): A Pandas DataFrame object constructed from the `nested_dict`
            with row index specified by `rows` and column headers specified by `columns`.
    """
    # Flatten the dictionary
    flat_dict = flatten_dict(nested_dict)

    # Convert the flattened dict to a DataFrame
    df = pd.DataFrame(list(flat_dict.items()), columns=['keys', 'values'])

    # Split keys into separate columns
    split_cols = df['keys'].str.split('|', expand=True)
    split_cols.rename(columns=dict(enumerate(index)), inplace=True)
    df = pd.concat([split_cols, df.drop('keys', axis=1)], axis=1)
    df['seq_len'] = df['seq_len'].astype(int)   # seq_len was str after split

    # Pivot the DataFrame to get desired structure
    df_pivot = df.pivot(index=rows, columns=columns, values='values')

    if csv_file is not None:
        print(f"Save data to: {csv_file}")
        df.to_csv(csv_file, index=True)

    return df_pivot
