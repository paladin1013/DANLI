# import logging
# import sys
# from llama_index import SimpleDirectoryReader
# from llama_index.indices.struct_store import GPTPandasIndex
# import pandas as pd

import os
import openai

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "https://localhost:7890"

# openai.organization = "org-p5ug2Pool5bdCna5a285PeCU"
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.proxy = {"http":"http://localhost:7890", "https":"https://localhost:7890"}


# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# df = pd.read_json("/home/yhgao/repositories/DANLI/teach-dataset/gpt_data/merged_output_test.json")

# index = GPTPandasIndex(df=df)

# query_engine = index.as_query_engine(
#     verbose=True
# )

# response = query_engine.query(
#     # "What is the city with the highest population?",
#     "What is required by the commander in the game with game_id bfa7505b440eadde_b257 and edh_num 8?",
# )
# response = query_engine.query(
#     # "What is the city with the highest population?",
#     "In which game session does the Commander requires the Follower to cook two slices of potatoes?",
# )

from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR('hkunlp/instructor-xl')
# sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
# instruction = "Represent the Science title:"
# embeddings = model.encode([[instruction,sentence]])
# print(embeddings)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
# sentences_a = [['Represent the Science sentence: ','Parton energy loss in QCD matter'], 
#                ['Represent the Financial statement: ','The Federal Reserve on Wednesday raised its benchmark interest rate.']]
# sentences_b = [['Represent the Science sentence: ','The Chiral Phase Transition in Dissipative Dynamics'],
#                ['Represent the Financial statement: ','The funds rose less than 0.5 per cent on Friday']]
# embeddings_a = model.encode(sentences_a)
# embeddings_b = model.encode(sentences_b)
# similarities = cosine_similarity(embeddings_a,embeddings_b)
# print(similarities)


GPT_RESULT_ROOT = "teach-dataset/gpt_data"
MERGED_OUTPUT = "merged_output.json"
EMBEDDINGS = "text_dialog_and_act_embeddings.npy"

with open(f"{GPT_RESULT_ROOT}/{MERGED_OUTPUT}", "r") as f:
    data_all = json.load(f)

# corpus = []
# for item in data_all:
#     corpus.append(["Represent the household work dialogue",item['text_dialog_and_act']])

# corpus_embeddings = model.encode(corpus)
# np.save(f"{GPT_RESULT_ROOT}/{EMBEDDINGS}", corpus_embeddings)

with open(f"{GPT_RESULT_ROOT}/{EMBEDDINGS}", "rb") as f:
    corpus_embeddings = np.load(f, allow_pickle=True)

while True:
    requirement = input("Please input the query requirement")
    query = [["Represent the household work requirement", requirement]]
    query_embeddings = model.encode(query)
        
    similarities = cosine_similarity(query_embeddings,corpus_embeddings)
    retrieved_doc_ids = similarities.reshape(-1).argsort()[-5:][::-1]

    for k, id in enumerate(retrieved_doc_ids):
        print(f"================== Closest: {k} ==================")
        print(data_all[id]['text_dialog_and_act'])