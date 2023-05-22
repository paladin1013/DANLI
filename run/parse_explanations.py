import os
import openai
from model.data.memory_manager import TaskMemoryManager
import logging
from model.model.gpt_api import GPTAPI
import json
from model.utils.format_utils import danli2gpt


manager = TaskMemoryManager(memory_split="all", data_root_dir="teach-dataset", log_level=logging.DEBUG)
gpt4_api = GPTAPI(model="gpt4")
gpt_sessions = gpt4_api.load_response_jsonl("teach-dataset/gpt_data/subgoal_explanation/subgoal_explainations_prompt_train_out1.jsonl")
results = []
manager.load_memory()
correct_game_ids = []
empty_game_ids = [] # Does not have valid gpt responses
incomplete_game_ids = [] # Does not have explanation for all subgoals
parsing_error_game_ids = [] # Error when parsing gpt response

for gpt_session in gpt_sessions.values():
    if gpt_session["error"] is not None:
        print(f"[{gpt_session['id']}] GPT does not have a valid response: {gpt_session['error']}")
        empty_game_ids.append(gpt_session["id"])
        continue
    try:
        game_id, subgoals, explanations = manager.parse_subgoal_explanations(gpt_session, ignore_invalid=False)
    except ValueError as e:
        print(f"[{gpt_session['id']}] Parsing error: {e}")
        parsing_error_game_ids.append(gpt_session["id"])
        continue
        
    game_memory = manager.retrieve_game_memory(game_id)
    all_subgoals = [sg for edh_sg in game_memory["processed_subgoals"] for sg in edh_sg if sg[1] != "isPickedUp"]
    parsed_cnt = 0
    try:
        for gt_subgoal in all_subgoals:
            gt_subgoal_gpt = danli2gpt(gt_subgoal)
            while gt_subgoal_gpt != subgoals[parsed_cnt]:
                parsed_cnt += 1
    except IndexError:
        incomplete_game_ids.append(gpt_session["id"])
        print(f"[{gpt_session['id']}] Incomplete explanation of subgoal: {gt_subgoal}")
        continue
    print(f"[{gpt_session['id']}] Parsed successfully!")
    correct_game_ids.append(gpt_session["id"])

# Save the classification of game sessions to a json file
with open("teach-dataset/gpt_data/subgoal_explanation/explanation_validity.json", "w") as f:
    json.dump({"correct":correct_game_ids, "empty":empty_game_ids, "incomplete":incomplete_game_ids, "parsing_error":parsing_error_game_ids}, f)

print(f"correct: {len(correct_game_ids)}, empty: {len(empty_game_ids)}, incomplete: {len(incomplete_game_ids)}, parsing_error: {len(parsing_error_game_ids)}")
