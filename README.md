# Building a foundation classifier in 3 steps

### Step 1: Binary classification (~25s)
For each label, prompt an LLM to answer with YES or NO (1 token = 1 forward pass) if the company matches that label. The system prompt and company description messages KVs are caches, thus inferencing is reduced only to the name of the label and 1 token output per comparison. Using llama3.2:3b, this ~0.1s required per one classification. Further optimizations can be made using batched inference and broadcasting the KV cache along the batch (220) dimension - This way the inference is basically done using a single forward step.

### Step 2: Zero shot classification on pruned label pool (Instant)
The proposed labels from the previous step are prompted to a bigger LLM to select the most relevant ones. Using llama3-70b-8192 on Groq API calls.

### Step 3: Scaling on bilions
Results obtained after the first N samples are now a synthetic-labeled training set for multi-label classification => Finetune any pretrained sequence classifier.

### Refinement:
For samples with high multilabel assignment, repool the result one again with step 2.

### Validation:
Validate a subset with larger models on Groq.