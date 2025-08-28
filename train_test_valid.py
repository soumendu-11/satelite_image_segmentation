# =========================
# Imports
# =========================
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
import json

# =========================
# Load dataframe
# =========================
df_final = pd.read_csv('df_final_1.csv')

# =========================
# Function definition
# =========================
def create_stratified_datasets(df: pd.DataFrame, text_col: str, label_col: str,
                               train_size: float = 0.8, valid_size: float = 0.1,
                               test_size: float = 0.1, random_state: int = 42):
    """
    Splits a dataframe into stratified train, validation, and test sets,
    converts them to Hugging Face Dataset objects, and saves them as CSV files.

    Args:
        df (pd.DataFrame): Input dataframe.
        text_col (str): Name of the text column.
        label_col (str): Name of the label column.
        train_size (float): Proportion of training data.
        valid_size (float): Proportion of validation data.
        test_size (float): Proportion of test data.
        random_state (int): Random seed.

    Returns:
        train_dataset, valid_dataset, test_dataset: Hugging Face Dataset objects
    """
    assert train_size + valid_size + test_size == 1.0, "Train + valid + test sizes must sum to 1."

    # Step 1: Split into train and temp (valid+test)
    train_data, temp_data = train_test_split(
        df,
        test_size=(1 - train_size),
        stratify=df[label_col],
        random_state=random_state
    )

    # Step 2: Split temp into validation and test
    relative_test_size = test_size / (valid_size + test_size)  # proportion relative to temp
    valid_data, test_data = train_test_split(
        temp_data,
        test_size=relative_test_size,
        stratify=temp_data[label_col],
        random_state=random_state
    )

    # Step 3: Rename columns to 'text' and 'label'
    def rename_cols(df_subset):
        return df_subset[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'}).reset_index(drop=True)

    train_data = rename_cols(train_data)
    valid_data = rename_cols(valid_data)
    test_data = rename_cols(test_data)

    # Step 4: Save as CSV
    train_data.to_csv('train.csv', index=False)
    valid_data.to_csv('val.csv', index=False)
    test_data.to_csv('test.csv', index=False)

    # Step 5: Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_data)
    valid_dataset = Dataset.from_pandas(valid_data)
    test_dataset = Dataset.from_pandas(test_data)

    return train_dataset, valid_dataset, test_dataset


def save_label_mapping(df: pd.DataFrame, label_col: str, output_file: str = "label_mapping.json"):
    """
    Creates and saves a label2id and id2label mapping as JSON.

    Args:
        df (pd.DataFrame): Input dataframe.
        label_col (str): Column name containing labels.
        output_file (str): Path to save the JSON mapping.
    """
    unique_labels = sorted(df[label_col].unique())
    label2id = {str(label): idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: str(label) for idx, label in enumerate(unique_labels)}

    mapping = {
        "label2id": label2id,
        "id2label": id2label
    }

    with open(output_file, "w") as f:
        json.dump(mapping, f, indent=4)

    print(f"Label mapping saved to {output_file}")


# =========================
# Example usage
# =========================
train_dataset, valid_dataset, test_dataset = create_stratified_datasets(
    df=df_final,
    text_col='Cleaned_Text',
    label_col='Label'
)

# Save label mapping
save_label_mapping(df_final, label_col='Label')

# Optional: check sizes
print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(valid_dataset)}")
print(f"Test size: {len(test_dataset)}")
