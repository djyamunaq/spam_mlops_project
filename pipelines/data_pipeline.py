import pandas as pd
from sklearn.model_selection import train_test_split
from prefect import flow, task

@task
def fetch_data(url: str) -> pd.DataFrame:
    """Fetches the raw data from a URL."""
    return pd.read_csv(url, sep='\t', names=['label', 'message'])

@task
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the raw data."""
    df = df.drop_duplicates()
    # Standardize labels: 'ham' -> 0, 'spam' -> 1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df = df.dropna()

    return df

@task
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates simple features from the text."""
    df['message_length'] = df['message'].apply(len)
    df['num_capitals'] = df['message'].apply(lambda x: sum(1 for c in x if c.isupper()))

    return df

@task
def split_data(df: pd.DataFrame) -> tuple:
    """Splits data into train, validation, and test sets."""
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42, stratify=train_df['label']) # 0.25 * 0.8 = 0.2
    return train_df, val_df, test_df

@flow(name="spam_data_pipeline")
def data_pipeline(url: str, output_path: str = "./data/processed/"):
    """Main Prefect flow for the data pipeline."""
    # Execute tasks
    raw_df = fetch_data(url)
    clean_df = clean_data(raw_df)
    featured_df = create_features(clean_df)
    train_df, val_df, test_df = split_data(featured_df)

    # Save the processed data
    train_df.to_csv(f"{output_path}/train.csv", index=False)
    val_df.to_csv(f"{output_path}/val.csv", index=False)
    test_df.to_csv(f"{output_path}/test.csv", index=False)

    print("Data pipeline finished successfully!")

    return train_df, val_df, test_df


if __name__ == "__main__":
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    data_pipeline(data_url)