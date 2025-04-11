import polars as pl
import time
import torch
from transformers import AutoTokenizer, ModernBertModel
from tqdm import tqdm
import numpy as np

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(100)  # optional, adjusts table width


def analyze_labels() -> None:
    labels = pl.read_csv("original_dataset/labels.csv")

    labels = labels.with_columns(
        pl.col("label").str.strip_chars().str.to_lowercase()
    )
    labels = labels.with_columns(
        pl.col("label").str.split(" ")
    )

    words = labels.explode("label")
    word_counts = (
    words.group_by("label")
    .agg(pl.count().alias("count"))
    .filter(pl.col("label") != "and")  # remove the row where label is "and"
    .sort("count", descending=True)
)
   
    print(word_counts.head(20))
    print(len(word_counts))

def has_common_substring(a: str, b: str, min_len: int = 4) -> bool:
    """
    Returns True if strings a and b share any substring of at least min_len characters.
    Uses a sliding window and set intersection for efficiency.
    """
    # If either string is shorter than min_len, no common substring of length min_len can exist
    if len(a) < min_len or len(b) < min_len:
        return False

    # Generate all substrings of length min_len from string `a`
    substrings_a = {a[i:i + min_len] for i in range(len(a) - min_len + 1)}
    
    # Check if any of those substrings are in string `b`
    for i in range(len(b) - min_len + 1):
        if b[i:i + min_len] in substrings_a:
            return True

    return False

def check_if_niches_in_labels():
    data = pl.read_csv("original_dataset/data.csv")
    labels = pl.read_csv("original_dataset/labels.csv")['label'].to_list()

    # remove common words from labels
    checking_words = [" with",  " and", " services", "manufacturing", "production", "construction", "residential", "installation", "commercial", " anagement", "processing", "health", "cleaning", "consulting", "application"]
    for idx in range(len(labels)):
        labels[idx] = labels[idx].lower()  # Modify the label to lowercase
        for j in checking_words:
            if j in labels[idx]:
                labels[idx] = labels[idx].replace(j, "")  # Modify in place
    for idx in range(len(labels)):
        labels[idx] = labels[idx].strip()  # Remove any leading or trailing whitespace
    print(labels)
    return
    count = 0
    for i in data.iter_rows():
        niche = i[-1]
        for j in labels:
            if has_common_substring(niche.lower(), j.lower(), 20):
                print(niche + " | " + j)
                count+=1
        print(count)


def embed_companies():
    def sample_to_text(sample):
        text = ""
        for i in sample:
            if i is None:
                continue
            if isinstance(i, list) and len(i) > 0:
                text += " ".join(i)
            else:
                text += i
        return text.lower()
    
    torch._dynamo.config.suppress_errors = True
    df = pl.read_csv("original_dataset/data.csv")
    device = 'cuda'
    model = ModernBertModel.from_pretrained("answerdotai/ModernBERT-base").to(device)
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    # Tokenize the company names
    embeds = []
    with torch.no_grad():
        for idx, sample in tqdm(enumerate(df.iter_rows())):
            text = sample_to_text(sample)
            inp = tokenizer(text, return_tensors='pt')   
            output = model(**inp.to(device))
            cls_token = output.last_hidden_state[0, 0].cpu().detach().numpy()
            embeds.append(cls_token)

    df = df.with_columns(pl.Series("embedding", embeds))
    df.write_json("dataset/data_with_embeddings.json")


def plot_data():
    start = time.time()
    df = pl.read_json("dataset/labels_with_embeddings.json")
    print(time.time() - start)  
    
    x= pl.Series("embedding", values=df["embedding"])
    embeddings_col = np.array(df["embedding"].to_list())
    
    from flashml.tools import plot_tsne
    plot_tsne(embeddings_col)

plot_data()