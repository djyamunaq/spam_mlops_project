import pandas as pd
import zipfile
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
from prefect import flow, task

@task
def fetch_data(url: str) -> pd.DataFrame:
    """Fetches the raw data from a URL, handling ZIP files properly."""
    try:
        # If it's a direct CSV URL, try to read it directly
        if url.endswith('.csv'):
            return pd.read_csv(url, sep='\t', names=['label', 'message'])
        
        # If it's a ZIP file, we need to handle it differently
        elif url.endswith('.zip'):
            # Download the ZIP file
            response = requests.get(url)
            response.raise_for_status()
            
            # Extract the ZIP file in memory
            with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
                # Look for the data file (usually named 'SMSSpamCollection')
                file_list = zip_file.namelist()
                data_file = None
                
                # Prioritize files that look like our data
                for file_name in file_list:
                    if 'spam' in file_name.lower() or 'sms' in file_name.lower():
                        data_file = file_name
                        break
                
                # If no obvious data file found, use the first non-readme file
                if not data_file:
                    for file_name in file_list:
                        if 'readme' not in file_name.lower():
                            data_file = file_name
                            break
                
                if not data_file:
                    raise ValueError(f"No suitable data file found in ZIP. Files: {file_list}")
                
                # Read the data file from the ZIP
                with zip_file.open(data_file) as file:
                    return pd.read_csv(file, sep='\t', names=['label', 'message'])
        
        else:
            # Try direct read for other cases
            return pd.read_csv(url, sep='\t', names=['label', 'message'])
            
    except Exception as e:
        print(f"Error fetching data from {url}: {e}")
        raise
    
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
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Spam ratio: {clean_df['label'].mean():.2%}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    data_pipeline(data_url)