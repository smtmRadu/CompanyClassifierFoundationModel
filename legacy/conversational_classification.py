
import polars as pl
from ollama import chat
from flashml.tools.nlp import display_chat
import re

MODEL = "artifish/llama3.2-uncensored:latest"
TEMPERATURE = 0.01
KEEP_ALIVE = "1m"
HIGH_LEVEL_QUESTIONS = [
    ('agriculture', "Does the company provide services related to agriculture, farming, or livestock?"),
    ('construction', "Does the company offer construction, building, or installation services?"),
    ('manufacturing',"Is the company involved in manufacturing or production of goods?"),
    ('it&consulting', "Does the company provide professional, consulting, or technology-related services?"),
    
    ('health&safety', "Is the company focused on health, safety, or environmental services?"),
    ('realestate', "Does the company offer services related to real estate, property management, or land development?"),
    ('landscaping', "Is the company involved in outdoor, landscaping, or infrastructure-related services?"),
    ("arts&events&nonprofit","Does the company provide services in the arts, events, or non-profit sectors?"),
]

Q1 = "Is this company doing Services, Manufacturing, Construction, or Installation?"
Q2 = "What is the primary industry or domain this activity serves?"
Q3 = "What is the main function or output of this activity?"
Q4 = "Does this activity target a specific client type or scale?"
Q5 = "Is there a specific material, system, or focus involved?"

def extract_single_box_content(text):
    pattern = r'<box>(.*?)</box>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        raise "No box found in the answer"
    

def get_response_pool(sample) -> list[str]:
    description, business_tags, sector, category, niche = sample
    response_pool = []
    messages = [
                {
                    'role': 'system',
                    'content':f"You are a classification assistant helping to categorize company descriptions into specific labels. These is information related to a company: \
                        Description: {description},\
                        Business Tags (keywords about what the company does and offers): {business_tags},\
                        Sector (a high classification of the industry the company operates in): {sector},\
                        Category (a subdivision of the industry within it's sector): {category},\
                        Niche (the specialized focus of the company that's very specific): {niche}.\
                        The user will ask you a series of questions to build a pool of answers.\
                        Think about the question and provide detailed explanation for you answer. At the end provide a shorter answer (1-3 words) inserted inside the following tags <box></box>.\
                        Use the company description to determine the most appropriate answer from the options provided."
                }
            ]
    for q in [Q1, Q2, Q3, Q4, Q5]:
        messages.append({
            "role": "user",
            "content": q
        })
        response = chat(MODEL, messages, stream=False, keep_alive=KEEP_ALIVE, options={'temperature':TEMPERATURE})['message']['content']
        messages.append({
            'role':'assistant',
            'content': response
        })
        response_pool.append(extract_single_box_content(response))
    display_chat(messages)
    return response_pool
def main():
    data = pl.read_csv("dataset/data.csv")

    for sample in data.iter_rows():
        response_pool = get_response_pool(sample)
        print(response_pool)
        exit()

    # time.sleep(10)






if __name__ == "__main__":
    main()