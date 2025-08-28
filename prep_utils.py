
# =========================
# prep_utils.py
# =========================

import re
import pandas as pd
import plotly.express as px


# =========================
# Functions
# =========================
def clean_text(text: str) -> str:
    """
    Cleans text by removing HTML tags, non-alphanumeric characters,
    extra spaces/newlines, and trims whitespace.
    """
    if pd.isnull(text):
        return ""
    # Remove HTML/XML-like tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove non-alphanumeric characters (except punctuation and space)
    text = re.sub(r"[^a-zA-Z0-9.,;:()'\"!?% \n]", " ", text)
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    return text.strip()


def analyze_dataframe(csv_path: str, save_path: str = "df_final_1.csv"):
    """
    Reads a CSV, cleans the 'Text' column, prints and removes duplicates and nulls,
    performs analysis, saves the final cleaned dataframe, and plots distributions.
    """
    # Load dataframe
    df_final = pd.read_csv(csv_path)

    # Clean the text column
    df_final['Cleaned_Text'] = df_final['Text'].apply(clean_text)

    # Percentage of null/empty values in 'Cleaned_Text'
    null_percent = df_final['Cleaned_Text'].isnull().mean() * 100
    empty_percent = (df_final['Cleaned_Text'] == "").mean() * 100
    print(f"Percentage of null values in 'Cleaned_Text': {null_percent:.2f}%")
    print(f"Percentage of empty values in 'Cleaned_Text': {empty_percent:.2f}%")

    # Percentage of duplicate values in 'Cleaned_Text'
    duplicate_percent = df_final['Cleaned_Text'].duplicated().mean() * 100
    print(f"Percentage of duplicate values in 'Cleaned_Text': {duplicate_percent:.2f}%")

    # Remove duplicates based on 'Cleaned_Text'
    df_final.drop_duplicates(subset=['Cleaned_Text'], inplace=True)

    # Remove rows with null or empty 'Cleaned_Text'
    df_final = df_final[df_final['Cleaned_Text'].notnull() & (df_final['Cleaned_Text'] != "")]

    # Reset index
    df_final.reset_index(drop=True, inplace=True)

    # Preview
    print(df_final[['Text', 'Cleaned_Text']].head(10))

    # Token count per row
    df_final['token_count'] = df_final['Cleaned_Text'].apply(lambda x: len(str(x).split()))

    # Plot token distribution
    fig = px.histogram(df_final, x='token_count', nbins=50, title='Token Distribution in Cleaned_Text')
    fig.show()

    # Row with max tokens
    max_idx = df_final['token_count'].idxmax()
    max_tokens = df_final.loc[max_idx, 'token_count']

    print(f"\nRow with maximum tokens: {max_idx}")
    print(f"Maximum number of tokens: {max_tokens}")

    # Rows exceeding 512 tokens
    num_exceeding_512 = (df_final['token_count'] > 512).sum()
    percentage_exceeding_512 = (num_exceeding_512 / len(df_final)) * 100

    print(f"Percentage of rows with token count > 512: {percentage_exceeding_512:.2f}%")

    # =========================
    # NEW: Plot Label frequency distribution
    # =========================
    if 'Label' in df_final.columns:
        fig_label = px.histogram(
            df_final,
            x='Label',
            title='Label Frequency Distribution',
            color='Label',
            text_auto=True
        )
        fig_label.update_layout(xaxis_title="Label", yaxis_title="Count")
        fig_label.show()
    else:
        print("\n⚠️ Warning: No 'Label' column found in the dataframe. Skipping label distribution plot.")

    # Save final cleaned dataframe
    df_final.to_csv(save_path, index=False)
    print(f"\nCleaned dataframe saved as '{save_path}'")

    return df_final
