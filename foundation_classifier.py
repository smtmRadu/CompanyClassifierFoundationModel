import random
import polars as pl
from itertools import chain
from ollama import chat
from copy import deepcopy
from tqdm import tqdm
import re
from groq import Groq
from typing import Literal
import os
from dotenv import load_dotenv



class FoundationClassifier:
    def __init__(self, data, llm:Literal["ollama", "groq"]="ollama"):
        self.data = data
        self.labels = pl.read_csv("labels.csv")["label"].to_list()
        self.llm = llm
        self.llm_temperature = 0.5
        self.ollama_default_model = 'llama3.2:latest'
        self.groq_default_model = "llama3-70b-8192"
        load_dotenv()

        self.groq_api_key = os.environ.get("GROK_API_KEY")
        self.groq_client = Groq(api_key=os.environ.get("GROK_API_KEY"))


    def run(self, start, end):
        range_inclusive = range(start, end)
        for i in range_inclusive:
            row = self.data.row(i)
            if row[-1] is not None:
                print(f"Sample at index {i} already classified...")
                continue
            

            print(f"\n\nClassifying sample {i}. Classification running...")
            pool = deepcopy(self.labels)
            pool = self.perform_binary_decider(row, pool)
            final_pool = self.perform_final_pooling(row, pool)
            print(f"Classified with a pool of size {len(final_pool)}")

            self.data = self.data.with_columns([
                pl.when(pl.arange(0, self.data.height) == i)
                .then(final_pool)
                .otherwise(self.data["insurance_label"])
                .alias("insurance_label")
            ])
            self.data.write_json("predictions.json")


    def perform_binary_decider(self, company_row, answer_pool):
        SYSTEM_PROMPT = f'''You are an insurance taxonomy expert. Respond only with YES or NO (no elaboration) if the company fits the user-specified category.'''
        messages = [
            {
                'role':'system',
                'content':SYSTEM_PROMPT
            },
            {
                'role':'user',
                'content':f'''Company Description:{company_row[0]}
                Company Keywords: {company_row[1]} {company_row[2]} {company_row[3]} {company_row[4]}
                '''
            }   
        ]
        new_pool = []
        for answer in tqdm(answer_pool):
            messages.append({
                'role':'user',
                'content': f"{answer}?"
            })
            response = chat(self.ollama_default_model,messages, stream=False,keep_alive='1m', options={'temperature':0.0000001, 'num_predict':1})
            
            
            response_message = response['message']
            response_content = response_message['content']
            # messages.append(response_message)
            # display_chat(messages)
            # messages.pop()
            
            messages.pop()
            if response_content.lower() == "yes":
                new_pool.append(answer)
            elif response_content.lower() == "no":
                continue
            else:
                continue # for now i don't want my code to block here...
                print(f"This llm is stupid as fuck, it answered {response_content}")
                raise f"This llm is stupid as fuck, it answered {response_content}"
        return new_pool

    def perform_final_pooling(self, company_row, reduced_pool):
        # remove the system prompt
        messages = [
            {
                "role":'system',
                'content':'You are an insurance taxonomy expert. Given a company and a set of taxonomy choices, analyze which ones fit perfectly with to the company. After that, return your final answer (if only one like <box>answer</box> or if multiple like <box>answer1, answer2, answer3 ...</box>), within the <box> </box> tags.'
            },
            {
                'role':'user',
                'content':f'''Company Description:{company_row[0]}
                Business tags: {company_row[1]}
                Sector: {company_row[2]}
                Category: {company_row[3]}
                Niche:{company_row[4]}
                '''
            }
        ]
        messages.append(
            {
                'role':'user',
                'content':f"Answer Choices: {', '.join(reduced_pool)}"
            }
        )
        response = self.groq_client.chat.completions.create(messages=messages, model=self.groq_default_model, temperature=self.llm_temperature)
        response_content = response.choices[0].message.content 

        # messages.append(
        #     {
        #         'role':'assistant',
        #         'content':response_content
        #     }
        # )
        # display_chat(messages)
        matches = re.findall(r'<box>(.*?)</box>', response_content)

        final_pool = matches[-1].split(",")
        final_pool = [x.strip() for x in final_pool]
        return final_pool   

    def extract_answer_from_response_text(self, response_text:str):
        # Find all occurrences of text inside <box>...</box> tags
        matches = re.findall(r'<box>(.*?)</box>', response_text)

        if matches:
            return matches[-1]
        else:
            return None

if __name__ == "__main__":
    # Load file and create a new predictions file with insurance_label column
    
    # ON INITIALIZATION
    # df = pl.read_csv('data.csv')
    # predictions = df.with_columns(pl.lit(None).alias("insurance_label"))
    # predictions.write_json("predictions.json")

    predictions = pl.read_json("predictions.json")
    classifier = FoundationClassifier(predictions, "ollama").run(0, 100)

    # Note. At the full end of classification. ALl results must be rechecked if they belong to the labels file.
    # if not, they are checked if they match very close the ones in the file. Otherwise, they are discarded.
    