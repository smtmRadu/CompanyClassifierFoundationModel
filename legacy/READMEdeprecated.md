# Building a foundation classifier in 4 steps

### Step 1: Early pruning (Heuristic - Optional)
There are 5 important clusters that can be pruned by asking an llm if the company: 
- **is offering services?** 
- **is doing manufacuting?**
- **is making instalations?**
- **is related to constructions of any type?**
- **is doing production?**

This is a YES/NO answer. If the answer for any of these is NO, all labels that contain that "word" are removed from the label pool to reduce it's size drastically.

### Step 2: Grouping Strategy

The labels should be grouped based on their disimilarity (a simple solution is checking for word non-overlapping). For multi-type answer where (None of the above) choice is available for answer removals.

Initially (for the first phase of the tournament), a high quality handmade grouping can be used.

### Step 2: Tournament Selection

#### Single-type answer algorithm:

1. Early Pruning.
2. Group labels in N groups of size G (by disimilarity). G always divides N.
3. For each group, prompt an llm to select the label that fits the company best (if fails, the group is discarded).
4. Form a pool with all the answers and repeat until one label left.

#### Multi-type answer:
1. Early Pruning.
2. Group labels in N groups of size G (by disimilarity). G always divides N. Each group receives a (``None of the above``) choice.
3. For each group, prompt an llm to select the label that fits the company the most (if fails or if ``None of the above`` is selected, the group is discarded). If one answer is selected, it is placed in a special pool called ``Qualified pool``. The rest of the answers will join the ``Play-out pool``.
4. ``Play-out pool`` is passed through a Single-type answer algorithm (for simplicity), with higher G. Also, a multi-type LLM answer is allowed in prompting for efficiency. The results are moving in the ``Qualified pool``.
5. The ``Qualified pool`` at this point should have around 10 labels. This can be a zero-shot problem already for a big LLM.

### Step 3: Validation
Leverages the group size G and LLM model.

### Step 4: Scaling on bilions
Results obtained after the first N samples are now a training set for multi-label classification => Finetune any BERT-like model.