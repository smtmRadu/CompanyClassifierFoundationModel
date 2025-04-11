import random
from flashml.tools import shuffle_df
import polars as pl
from itertools import chain
from ollama import chat
from flashml.tools.nlp import display_chat
from copy import deepcopy
from tqdm import tqdm
import re
from flashml.tools import benchmark
import math
from functools import reduce
def generate_subsample_answers(available_labels:list[str], group_size=5) -> list[str]:
    # random.shuffle(available_labels)
    subgroups = []

    while len(available_labels) > 0:
        subset = [available_labels.pop(0)]
        
        for i in range(group_size - 1): # one already added at the beginning

            # try add a different type of label
            added_different_label = None
            #-------------------------------------------
            for i in available_labels:
                already_appended_words = set(chain.from_iterable(x.lower().split() for x in subset))
                this_new_label_words = i.lower().split()
                if not any(word in this_new_label_words for word in already_appended_words):
                    subset.append(i)
                    added_different_label = i
                    break
                    
            #------------------------------------------
            # if couldn't add a different type of label, just add the next one
            if added_different_label is not None:
                available_labels.remove(added_different_label)
            elif len(available_labels) > 0:
                subset.append(available_labels.pop(0))

        subgroups.append(subset)

    return subgroups

def assign_characters_to_answers(answer_sets:list[list[str]], shuffle_variants = True):
    '''
    Shuffle variants and keep A, B, C, D, E consistent
    '''
    new_list:list[list[tuple[str, str]]] = []

    for answer_set in answer_sets:
        options = []
        variants = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
        if shuffle_variants:
            random.shuffle(answer_set)
        for answer in answer_set:
            options.append((variants.pop(0), answer))

        new_list.append(options)

    return new_list

def extract_answer_from_response_text(response_text:str):
        # Find all occurrences of text inside <box>...</box> tags
        matches = re.findall(r'<box>(.*?)</box>', response_text)

        if matches:
            return matches[-1]
        else:
            return None

def perform_tournament_phase(company_row, answer_sets) -> list[str]:
    '''
    Perform tournament base classification, returns a reduced set of labels
    '''
    
    MODEL = 'llama3.2:latest'
    SYSTEM_PROMPT = f'''You are an insurance taxonomist expert that have to classify a company based on the details provided by the user. He gives you a set of answer choices, and you have to select the one that is the most appropriate for the company. Write your final answer like this: <box>A</box> or <box>B</box> or <box>C</box> or etc.'''
    USER_PROMPT = f"Company Description: {company_row[0]}. Business tags: {company_row[1]}. Sector: {company_row[2]}. Category: {company_row[3]}. Niche: {company_row[4]}. "
    
    the_new_pool = []

    for set_ in tqdm(answer_sets):
        ANSWERS = f"Answer Choices:\n{"\n".join(x+": "+y for x,y in set_)}"
        messages = [
            {
                'role':'system',
                'content':SYSTEM_PROMPT
            },
            {
                'role':'user',
                'content': USER_PROMPT + "\n" + ANSWERS
            }
        ]

        # First try
        response = chat(MODEL,messages, stream=False,keep_alive='1m', options={''})
        messages.append(response['message'])
        answer_letter = extract_answer_from_response_text(response['message']['content'])
        answer_label = next((tup for tup in set_ if tup[0] == answer_letter), None)

        # if it fails to answer, it seems like it is because there is no good answer to give, so we skip this group

        if answer_label is not None:
            the_new_pool.append(answer_label[1])

        # # if it continues to be stupid
        # if answer_label is None:
        #     messages.append(
        #         {
        #             'role':'user',
        #             'content':"You forgot to provide a final answer (even if none of the answers matches the company - in this case choose the one that is most appropriate) in the following format (with only the character corresponding to the answer within the tags): <box>A</box> or <box>B</box> or <box>C</box> or etc."
        #         }
        #     )
        #     response = chat(MODEL,messages, stream=False,keep_alive='1m')
        #     messages.append(response['message'])
        #     answer_letter = extract_answer_from_response_text(response['message']['content'])
        #     answer_label = next((tup for tup in set_ if tup[0] == answer_letter), None)
# 
        # if answer_label is None:
        #     display_chat(messages)
        #     raise "This AI is stupid as fuck"
        # 
# 
        # answer_label = answer_label[1]
# 
        # # messages.append({
        # #     'role':'user',
        # #     'content':answer_letter + ": " + answer_label
        # # })

        # the_new_pool.append(answer_label)

    return the_new_pool
       

def greatest_divisor_below_k(n, k):
    for d in range(k, 0, -1): 
        if n % d == 0:
            return d
    return None 

# Around 120 seconds per sample..
def main():
    MAX_GROUP_SIZE = 5
    data = pl.read_csv("original_dataset/data.csv")
    labels = pl.read_csv("original_dataset/labels.csv")["label"].to_list()


    
    for idx, sample in enumerate(data.iter_rows()):
        if idx != 7554:
            continue
        print(sample[0])
        new_pool = deepcopy(labels)
        # RUN TOURNAMENT
        
        while len(new_pool) > 1:
            GROUP_SIZE = greatest_divisor_below_k(len(new_pool), MAX_GROUP_SIZE + 1)
            if GROUP_SIZE == 1:
                GROUP_SIZE = MAX_GROUP_SIZE
                if len(new_pool) % GROUP_SIZE == 1:
                    GROUP_SIZE = 4
                if len(new_pool) % GROUP_SIZE == 1:
                    GROUP_SIZE = 3
                if len(new_pool) % GROUP_SIZE == 1:
                    GROUP_SIZE = 2
                if len(new_pool) % GROUP_SIZE == 1:
                    GROUP_SIZE = 6 # if that number sucks so much
                if len(new_pool) % GROUP_SIZE == 1:
                    GROUP_SIZE = 7 # if that number sucks so much

            print(f"POOL SIZE: {len(new_pool)} | GROUP SIZE: {GROUP_SIZE} | NO. GROUPS {len(new_pool)/GROUP_SIZE}")
            answer_sets = generate_subsample_answers(new_pool, group_size=GROUP_SIZE)
            answer_sets = assign_characters_to_answers(answer_sets, shuffle_variants=True)
            new_pool = perform_tournament_phase(sample,answer_sets)
            
            print()
            
        print(f"Company {idx+1}: {new_pool[0]}")

        break

if __name__ == '__main__':
    benchmark(main)
