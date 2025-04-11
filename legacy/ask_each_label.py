import polars as pl
from ollama import chat
from flashml.tools.nlp import display_chat
import re
from tqdm import tqdm
MODEL = "cogito:3b" # "gemma3:1b"# "artifish/llama3.2-uncensored:latest"
TEMPERATURE = 0.01
KEEP_ALIVE = "1m"


def extract_single_box_content(text):
    pattern = r'<box>(.*?)</box>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        print(f"No box found in the answer (received {text})")
        raise "err"
    

def get_response_pool(sample, labels:list) -> list[str]:
    description, business_tags, sector, category, niche = sample
    response_pool = []
    messages = [
                {
                    'role': 'system',
                    'content':f"You are a robust assistant used for new insurance taxonomy classification. These is information related to a company: \
                        Description: {description},\
                        Business Tags (keywords about what the company does and offers): {business_tags},\
                        Sector (a high classification of the industry the company operates in): {sector},\
                        Category (a subdivision of the industry within it's sector): {category},\
                        Niche (the specialized focus of the company that's very specific): {niche}.\
                        You must say either <box>YES</box> or <box>NO</box>."
                }
            ]
    
    for q in tqdm(labels):
        messages.append({
            'role':"user",
            "content": f"Does the company has anything related to {q}?"
        })
        response = chat(MODEL, messages, stream=False, keep_alive=KEEP_ALIVE, options={'temperature':TEMPERATURE})['message']['content']
        if extract_single_box_content(response) == "YES":
            response_pool.append(q)
        # messages.append({
        #     'role':'assistant',
        #     'content':response
        # })
        # display_chat(messages)
        messages.pop()
        # messages.pop()
    return response_pool

def main():
    print(f"Using model {MODEL}")
    labels = pl.read_csv("dataset/labels.csv")['label'].to_list()
    data = pl.read_csv("dataset/data.csv")

    for sample in data.iter_rows():
        response_pool = get_response_pool(sample, labels)
        print(response_pool)
        exit()

main()