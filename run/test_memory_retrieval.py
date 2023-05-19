# import logging
# import sys
# from llama_index import SimpleDirectoryReader
# from llama_index.indices.struct_store import GPTPandasIndex
# import pandas as pd

import os
import openai
from model.data.memory_manager import TaskMemoryManager
import logging
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "https://localhost:7890"

manager = TaskMemoryManager(memory_split="train", data_root_dir="teach-dataset", log_level=logging.DEBUG)
reply_str = """
Sure, here are the explanations for the subgoals from the dialogue:

[Dialogue] Robot: Hello. How can I assist you?
[Dialogue] Commander: Hi
[Dialogue] Commander: slice lettuce
[Dialogue] Robot: Sure. Knife location, please?
[Dialogue] Commander: on the cabinet on top of the microwave
[Dialogue] Robot: Thank you.
[Subgoal] 1. Manipulate(Slice, Lettuce)
<Explanation: The first task the commander gives to the robot is to slice the lettuce. Therefore, the initial subgoal is for the robot to slice the lettuce using the knife.>

[Dialogue] Robot: Lettuce sliced.
[Dialogue] Robot: Next?
[Dialogue] Commander: slice a tomato
[Dialogue] Robot: OK. Tomato location?
[Dialogue] Commander: in the fridge
[Dialogue] Robot: Thank you.
[Subgoal] 2. Manipulate(Slice, Tomato)
<Explanation: The commander's next task is to have a tomato sliced. Thus, the next subgoal is to slice the tomato.>

[Dialogue] Robot: Tomato sliced.
[Dialogue] Robot: Next?
[Dialogue] Commander: cook a slice of potato
[Dialogue] Robot: Sure
[Subgoal] 3. Manipulate(Cook, PotatoSliced)
<Explanation: Here, the commander instructs the robot to cook a slice of potato. The subgoal in this instance is to cook the already sliced potato.>

[Dialogue] Robot: Potato cooked and sliced. Next?
[Dialogue] Commander: add all sales components on the plate
[Dialogue] Robot: Sure
[Subgoal] 4. Place(Plate, DiningTable)
<Explanation: The commander requests the robot to prepare a plate with the sliced ingredients. The subgoal, in this case, is to have the plate ready on the dining table.>

[Dialogue] Robot: How many slices of lettuce for the plate?
[Subgoal] 5. Place(LettuceSliced, Plate)
<Explanation: Now that the plate is ready, the next task is to place the sliced lettuce on it, which is what this subgoal represents.>

[Dialogue] Commander: one
[Dialogue] Robot: OK
[Dialogue] Robot: How many slices of tomato for the plate?
[Subgoal] 6. Place(TomatoSliced, DiningTable)
<Explanation: This subgoal seems to be a mistake, as the robot should place the sliced tomato on the plate, not the dining table.>

[Subgoal] 7. Place(TomatoSliced, Plate)
<Explanation: This is the correct subgoal as per the commander's instructions. The robot is to place the sliced tomato on the plate.>

[Dialogue] Robot: How many potato slices?
[Dialogue] Commander: 2
[Dialogue] Commander: one tomato, two potatoes
<Explanation: According to the commander's instructions, the robot needs to place one slice of tomato and two slices of potato on the plate. This forms the final configuration of the dish as per the dialogue.>"""
prompt_str = """
Please add brief explanations of each subgoal according to the previous dialog:

[Dialogue] Robot: Hello. How can I assist you?
[Dialogue] Commander: Hi
[Dialogue] Commander: slice lettuce
[Dialogue] Robot: Sure. Knife location, please?
[Dialogue] Commander: on the cabinet on top of the microwave
[Dialogue] Robot: Thank you.
[Dialogue] Commander: cooker
[Dialogue] Robot: Thank you.
[Subgoal] 1. Manipulate(Slice, Lettuce)
[Dialogue] Robot: Lettuce sliced.
[Dialogue] Robot: Next?
[Dialogue] Commander: slice a tomato
[Dialogue] Robot: OK. Tomato location?
[Dialogue] Commander: in the fridge
[Dialogue] Robot: Thank you.
[Subgoal] 2. Manipulate(Slice, Tomato)
[Dialogue] Robot: Tomato sliced.
[Dialogue] Robot: Next?
[Dialogue] Commander: cook a slice of potato
[Dialogue] Robot: Sure
[Subgoal] 3. Manipulate(Cook, PotatoSliced)
[Dialogue] Robot: Potato cooked and sliced. Next?
[Dialogue] Commander: add all sales components on the plate
[Dialogue] Robot: Sure
[Subgoal] 4. Place(Plate, DiningTable)
[Dialogue] Robot: How many slices of lettuce for the plate?
[Subgoal] 5. Place(LettuceSliced, Plate)
[Dialogue] Commander: one
[Dialogue] Robot: OK
[Dialogue] Robot: How many slices of tomato for the plate?
[Subgoal] 6. Place(TomatoSliced, DiningTable)
[Subgoal] 7. Place(TomatoSliced, Plate)
[Dialogue] Robot: How many potato slices?
[Dialogue] Commander: 2
[Dialogue] Commander: one tomato, two potatoes

Here are some example explanations of three subgoals:
[Dialogue] Robot: hello what is my task
[Dialogue] Commander: Today, you'll be preparing breakfast.
[Dialogue] Commander: First, make coffee.
[Dialogue] Commander: Put the Place the coffee on the table.
[Subgoal] 1. Place(Mug, CoffeeMachine) <Explanation:The commander requires the robot to make a coffee as a part of breakfast. The robot should first put a mug inside the coffee machine.>
[Subgoal] 2. Manipulate(FillWithCoffee, Mug) <Explanation: After the mug is placed inside the coffee machine, it can be filled with coffee afterwards.>
[Subgoal] 3. Place(Coffee, DiningTable) <Explanation: We use Coffee to represent the mug which is filled with coffee. It should be placed on the dining table according to the commander's instruction.>

Please start to explain from the beginning of dialogue and subgoals. Please add <Explanation> after each [Subgoal] while also keep every [Dialogue]."""
gpt_session = {"id": "2b489b344a9ee00e_9717", "responses": [reply_str], "prompts": [prompt_str]}

game_id, subgoals, explanations = manager.parse_subgoal_explanations(gpt_session)
print(game_id, subgoals, explanations)
game_memory = manager.retrieve_game_memory(game_id)
all_subgoals = [sg for edh_sg in game_memory["processed_subgoals"] for sg in edh_sg if sg[1] != "isPickedUp"]
print(len(subgoals), len(all_subgoals))
# manager.process_memory()
# manager.process_memory()
# game_id = "062836eb156ac8b8_f3de"
# game_id = "dd9a71ec1af06961_30d8"
# game_id = "2b489b344a9ee00e_9717"
# manager.load_memory()

# game_memory = manager.retrieve_game_memory(game_id)
# explanation_prompt = manager.generate_explaination_prompt(game_memory)
# print(explanation_prompt)

# synthesized_dialogue, synthesized_dialogue_and_subgoals = manager.synthesize_edh_sessions(game_memory)
# print(synthesized_dialogue_and_subgoals)
# task = manager.query_task_memory("Make sandwich", top_k=5)
# print(task[0])
# manager.query_task_memory("Wash dishes", top_k=5)
# manager.query_task_memory("Wash apples")


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

# from InstructorEmbedding import INSTRUCTOR
# model = INSTRUCTOR('hkunlp/instructor-xl')
# # sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
# # instruction = "Represent the Science title:"
# # embeddings = model.encode([[instruction,sentence]])
# # print(embeddings)
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import json


# GPT_RESULT_ROOT = "teach-dataset/gpt_data"
# MERGED_OUTPUT = "merged_output.json"
# EMBEDDINGS = "text_dialogue_and_act_embeddings.npy"

# with open(f"{GPT_RESULT_ROOT}/{MERGED_OUTPUT}", "r") as f:
#     data_all = json.load(f)

# # corpus = []
# # for item in data_all:
# #     corpus.append(["Represent the household work dialogueue",item['text_dialogue_and_act']])

# # corpus_embeddings = model.encode(corpus)
# # np.save(f"{GPT_RESULT_ROOT}/{EMBEDDINGS}", corpus_embeddings)

# with open(f"{GPT_RESULT_ROOT}/{EMBEDDINGS}", "rb") as f:
#     corpus_embeddings = np.load(f, allow_pickle=True)

# while True:
#     requirement = input("Please input the query requirement")
#     query = [["Represent the household work requirement", requirement]]
#     query_embeddings = model.encode(query)
        
#     similarities = cosine_similarity(query_embeddings,corpus_embeddings)
#     retrieved_doc_ids = similarities.reshape(-1).argsort()[-5:][::-1]

#     for k, id in enumerate(retrieved_doc_ids):
#         print(f"================== Closest: {k} ==================")
#         print(data_all[id]['text_dialogue_and_act'])