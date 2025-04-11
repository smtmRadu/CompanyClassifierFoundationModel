
import polars as pl
from ollama import chat
from flashml.tools.nlp import display_chat
import re
from copy import deepcopy
STOP_WORDS = ["","with", "and"] # also non from nonprofit/nonalcoholic, e comes from e commerce

def create_label_mapping(labels_list):
    label_mapping = {}
    for label in labels_list:
        label_name = label.lower()
        label_name = re.sub(r"[^\w\s]", " ", label_name)
        label_tags = label_name.split()
        cleaned_tags = []
        # remove stop words 
        for tag in label_tags:
            if tag not in STOP_WORDS:
                for sw in ["multi", "single", "non"]:
                    if sw in tag:
                        # remove stopword in tag
                        tag = tag.replace(sw, "")

                # remove plural
                if tag.endswith("s") and len(tag) > 4: # (gas, GLASS, CANVAS, BUSINESS) # from this point they must find a match anyways
                    tag = tag[:-1]
                
                if tag.endswith("ing"):
                    tag = tag[:-3]

                if tag != "" and len(tag) > 1:
                    cleaned_tags.append(tag)

            # make stemming out of the tags
        # make stemming out of the tags

        label_mapping[label] = cleaned_tags
        

    return label_mapping

def main():
    data = pl.read_csv("dataset/data.csv")
    labels = pl.read_csv("dataset/labels.csv")["label"].to_list()
    label_mapping_tags = create_label_mapping(labels)

    for idx, sample in enumerate(data.iter_rows()):

        if idx != 23:
            continue

        label_mapping_scores = {}
        text = " ".join(sample).lower()

        for label_kv in label_mapping_tags.items():
            label = label_kv[0]
            label_tags = label_kv[1]
            score = 0
            for each in label_tags:
                if each in text:
                    score += 1
            
            
            label_mapping_scores[label] = score

        # sort the scores in descending order
        sorted_scores = sorted(label_mapping_scores.items(), key=lambda x: x[1], reverse=True)
        # select only the ones with scores + 2
        sorted_scores = [x for x in sorted_scores if x[1] >= 2]
        print("TEXT:")
        print(sample[0])
        print("OPTIONS:")
        print([x[0] for x in sorted_scores])
      
        # now try to find if the stemmed tags of the labels can be found in the text

        

    # time.sleep(10)






if __name__ == "__main__":
    main()