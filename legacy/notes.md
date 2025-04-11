## Problem Definition

1. Description: summary of the company services/operations/insights etc.
2. Business tags: keywords for what the company does and offers
3. Sector: high-level classification of the industry
(Sevices, Manufacturing, Agriculture etc.)
4. Category: Subdivision of the industry within the sector
(e.g. Retail, Wholesale, Distribution etc.)
5. Niche: Specialized focus of the company, highly specific

Note: there are samples without descriptions.

### Plan 1
The labels can be classified in certain ways. We can categorize them similar to how Akinator does in a certain way.

Solution. For each top 5 (services, manufacturing, installation, construction, production) ask for each separetely:

Does the company offers services?
Does the company offers manufacturing?
Does the company offers installation?
Does the company offers construction?
Does the company offers production?

And for the rest work like:

Is it a comercial company?
Is it a residential company?
Is it a management company?
Is it a procesing company?
Is it a consulting company?

And for the rest of the things:

Does the company mentions anything related to health?
Does the company have any linking to animal?

etc...

From that, we collect for which labels the llms responded yes. Then from that pool we check which labels have all the
words inside the pool and that's it.


### Plan 2

Generate multiple instances descriptions for every label. Encode then in a vector space and compare cosine similarity of each sample to the embedded representation of the label. The label with the highest similarity is the one that the company is most likely to match.


### Plan 3
Agentic classification
The classification is tournament based. Create subsets of different labels (as different as possible to allow easier selection).

# Methods for validation

After we assign labels we apply the following tricks:

- Ask the same model (or another model, a big one to be honest) to tell if the company is matching that assign label.
- Ask different models to classify zero shot the company and check, and also check zero shot but from a smaller
subset of the labels (choosing labels that are similar to the assigned label at inference). You should do this in a
continual manner like:
"Which of the following label would you assign to the company? ANSWER, What do you think about these ones? ANSWER. ...

- Check for consistency comparing to other models. Check how similar are the results from model to model.... 