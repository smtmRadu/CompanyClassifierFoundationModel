import polars as pl
from concurrent.futures import ThreadPoolExecutor
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import torch._dynamo.config
from transformers import AutoTokenizer, ModernBertModel
import torch
import torch._dynamo
from flashml.tools.nlp import *
from tqdm import tqdm
import numpy as np
from groq import Groq
from ollama import chat
import os
import json
from dotenv import load_dotenv


def get_wikipedia_text(name):
    # Construct the initial search query by appending "wikipedia" to the name
    query = f"{name} wikipedia"
    all_urls = []
    
    # Get multiple Wikipedia URLs
    try:
        for url in search(query, num_results=5):  # Limiting to 5 results for efficiency
            if "wikipedia.org" in url:
                all_urls.append(url)
        if not all_urls:
            return "No Wikipedia page found for this search."
    except Exception as e:
        return f"Error during search: {str(e)}"

    # Process URLs until we get sufficient text or run out of options
    for wikipedia_url in all_urls:
        try:
            response = requests.get(wikipedia_url)
            response.raise_for_status()  # Check for HTTP errors
            
            # Parse the page with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract all text from the main content area
            content = soup.find(id="mw-content-text")
            if not content:
                continue  # Move to next URL if no content found
            
            # Get all paragraph text
            paragraphs = content.find_all('p')
            full_text = "\n".join([para.get_text() for para in paragraphs])
            
            # Check if text length is sufficient
            if full_text and len(full_text) >= 1000:
                return full_text
            # If text exists but is too short, continue to next URL
            
        except requests.RequestException as e:
            return f"Error fetching Wikipedia page: {str(e)}"
        except Exception as e:
            return f"Error parsing Wikipedia page: {str(e)}"
    
    # If we've gone through all URLs and still don't have enough text
    if not full_text:
        raise "BRO WTF"
    return full_text

def labels_with_wiki_information():
    print("extracting wikipedia data")
    # Read the CSV and add an "info" column with default value
    labels = pl.read_csv("original_dataset/labels.csv").with_columns(
        pl.lit("none").alias("info")
    )
    
    # Extract the labels column as a Python list for processing
    label_list = labels["label"].to_list()
    
    # Use ThreadPoolExecutor to fetch Wikipedia text in parallel
    with ThreadPoolExecutor(max_workers=64) as executor:
        # Map get_wikipedia_text to each label in parallel
        results = list(executor.map(get_wikipedia_text, label_list))
    
    # Update the "info" column with the results
    labels = labels.with_columns(
        pl.Series("info", results)
    )
    
    labels.write_csv("dataset/labels_with_info.csv")





def write_to_disk(label_name, response_text):
    entry = {
        "label": label_name,
        "response": response_text
    }
    with open("insurance_taxonomy_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def generate_text_for_label(label_name):
    groq_client = Groq(api_key=os.environ.get("GROK_API_KEY"))
    messages = [
        {
            'role':'system',
            'content':"Your will offer the user a detailed description for the category the user gives to you, from the perspective of insurance taxonomy. You will only write the description and nothing more than that."
        },
        {
            "role":'user',
            "content": f"{label_name}"
        }
    ]
    response = groq_client.chat.completions.create(messages=messages, model="meta-llama/llama-4-scout-17b-16e-instruct")
    response = response.choices[0].message.content
    write_to_disk(label_name, response)
    return response

def labels_with_ai_information():
    print("generating detailed explanations of labels...")
    # Read the CSV and add an "info" column with default value
    labels = pl.read_csv("original_dataset/labels.csv").with_columns(
        pl.lit("none").alias("info")
    )
    
    # Extract the labels column as a Python list for processing
    label_list = labels["label"].to_list()
    
    results = []
    for idx, label in tqdm(enumerate(label_list)):
        results.append(generate_text_for_label(label))

    labels = labels.with_columns(
        pl.Series("info", results)
    )
    
    labels.write_csv("dataset/labels_with_ai_info.csv")



def assign_embedding_value_to_label_info():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCHDYNAMO_VERBOSE"] = "0"
    df = pl.read_csv("dataset/labels_with_ai_info.csv")    
    torch._dynamo.config.suppress_errors = True
    print("preprocessing")

    lowercase(df, "info")
    lowercase(df, "label")
    # replace_punctuation(df, "info", "")
    # replace_urls(df, "info", "")
    # replace_numbers(df, "info", "")
    # remove_double_spacing(df, "info")

    device = 'cuda'
    model = ModernBertModel.from_pretrained("answerdotai/ModernBERT-base").to(device)
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    embeddings = []
    with torch.no_grad():
        for idx, row in tqdm(enumerate(df.iter_rows())):
            text = " ".join(row)
            enc = tokenizer(text, return_tensors='pt')
            result = model(**enc.to(device))
            cls_token = result.last_hidden_state[0, 0]
            # append the emeddbing to the embedding column
            embeddings.append(cls_token.cpu().detach().numpy())


    df = df.with_columns(pl.Series("embedding", embeddings))
    df = df.drop("info")
    df.write_json("dataset/labels_with_ai_embeddings.json")

def sort_labels_by_embedding_similarity(labels_df: pl.DataFrame, target_embedding: list[float]) -> pl.DataFrame:
    # Convert all embeddings to a numpy array
    embeddings = np.vstack(labels_df['embedding'].to_list())
    target = np.array(target_embedding)

    # Compute cosine similarities
    dot_products = embeddings @ target
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target)
    cosine_similarities = dot_products / norms

    # Add cosine_similarity column and sort
    labels_df = labels_df.with_columns(
        pl.Series(name="cosine_similarity", values=cosine_similarities)
    )

    return labels_df.sort("cosine_similarity", descending=True)

def test_out():
    torch._dynamo.config.suppress_errors = True
    data = pl.read_csv("original_dataset/data.csv")
    labels = pl.read_json("dataset/labels_with_ai_embeddings.json")

    device = 'cuda'
    model = ModernBertModel.from_pretrained("answerdotai/ModernBERT-base").to(device)
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    with torch.no_grad():
        for idx, sample in enumerate(data.iter_rows()):
            sample_text = " ".join(sample[1:]).lower()
            inp = tokenizer(sample_text, return_tensors='pt')   
            output = model(**inp.to(device))
            cls_token = output.last_hidden_state[0, 0].cpu().detach().numpy()
            ding = sort_labels_by_embedding_similarity(labels, cls_token)
            print(ding)
            exit()



def generate_embeddings_for_labels():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCHDYNAMO_VERBOSE"] = "0"
    df = pl.read_csv("original_dataset/labels.csv")    
    torch._dynamo.config.suppress_errors = True

    device = 'cuda'
    model = ModernBertModel.from_pretrained("answerdotai/ModernBERT-base").to(device)
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    embeddings = []
    with torch.no_grad():
        for idx, row in tqdm(enumerate(df.iter_rows())):
            enc = tokenizer(row[0], return_tensors='pt')
            result = model(**enc.to(device))
            cls_token = result.last_hidden_state[0, 0]
            # append the emeddbing to the embedding column
            embeddings.append(cls_token.cpu().detach().numpy())


    df = df.with_columns(pl.Series("embedding", embeddings))
    df.write_json("dataset/labels_with_embeddings.json")

if __name__ == '__main__':
    load_dotenv()

    generate_embeddings_for_labels()