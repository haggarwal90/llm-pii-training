import datasets
import numpy as np
from transformers import AutoTokenizer
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

nlp_p_t = pipeline('ner', model='meta-llama/Llama-2-7b-hf', tokenizer=tokenizer)

example = 'Bill Gates is the founder of Delhi'

ner_results = nlp_p_t(example)
