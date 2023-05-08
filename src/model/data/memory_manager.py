"""A class to load, process, store and recall memory in raw text"""

import os
import sys
import json
from collections import defaultdict
import re
import logging
from InstructorEmbedding import INSTRUCTOR
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity


class TaskMemoryManager:
    def __init__(self, memory_split: str, data_root_dir: str, log_level=logging.INFO):
        if memory_split in ["train", "valid_seen", "valid_unseen"]:
            self.memory_split = memory_split
        else:
            raise ValueError(
                f"memory_split should be one of 'train', 'valid_seen', 'valid_unseen', but got {memory_split}."
            )
        self.data_root_dir = data_root_dir
        self.logger = logging.getLogger("TaskMemoryManager")
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.setLevel(log_level)
        self.model = None

    def process_memory(self):
        """Load task dialogue and groundtruth subgoals from raw edh files and save them to a memory json file"""
        edh_data = defaultdict(dict)
        data_dir = f"{self.data_root_dir}/edh_instances/{self.memory_split}"
        for path in os.listdir(data_dir):
            self.logger.debug(f"loading edh file name: {path}")
            with open(f"{data_dir}/{path}", "r") as f:
                temp_data = json.load(f)
            edh_num = int(re.match(".+\.edh(\d+)", temp_data["instance_id"]).group(1))
            new_edh_session = {}

            new_edh_session["future_subgoals"] = []
            for k in range(len(temp_data["future_subgoals"]) // 2):
                action = temp_data["future_subgoals"][2 * k]
                if action == "Navigate":
                    # Exclude navigation in subgoal planning. Whether to navigate should be done when completing subgoals
                    continue
                else:
                    new_edh_session["future_subgoals"].append(
                        [action, temp_data["future_subgoals"][2 * k + 1]]
                    )

            new_edh_session["dialog_history"] = []
            for dialog in temp_data["dialog_history_cleaned"]:
                role = dialog[0].replace("Driver", "Robot")
                new_edh_session["dialog_history"].append([role, dialog[1]])

            edh_data[temp_data["game_id"]][edh_num] = new_edh_session

        game_data_sorted = []

        for game_id, edh_session in edh_data.items():
            self.logger.debug(f"Processing game id {game_id}")
            game_session_sorted = {}
            game_session_sorted["game_id"] = game_id
            game_session_sorted["edh_nums"] = sorted(edh_session.keys())
            temp_subgoals = []
            temp_dialog = []
            prev_dialog_len = 0
    
            for edh_num in game_session_sorted["edh_nums"]:
                temp_subgoals.append(edh_session[edh_num]["future_subgoals"])
                temp_dialog.append(
                    edh_session[edh_num]["dialog_history"][prev_dialog_len:]
                )
                prev_dialog_len = len(edh_session[edh_num]["dialog_history"])
        
            game_session_sorted["edh_complete"] = self.update_holding_items(temp_subgoals)
            game_session_sorted["future_subgoals"] = temp_subgoals
            game_session_sorted["dialog_history"] = temp_dialog

            game_data_sorted.append(game_session_sorted)

        with open(
            f"{self.data_root_dir}/processed_memory/game_memory_{self.memory_split}.json",
            "w",
        ) as f:
            json.dump(game_data_sorted, f)
            
        self.task_memory = game_data_sorted

        self.memory_embeddings = self.calc_embeddings(self.task_memory)
        np.save(
            f"{self.data_root_dir}/processed_memory/embeddings_{self.memory_split}",
            self.memory_embeddings,
        )

    def update_holding_items(self, game_subgoals: List[List[List[str]]]):
        # subgoals_with_inventory = []
        holding_item = "Empty"
        edh_is_complete = True
        for edh_subgoals in game_subgoals:
            # edh_subgoals_with_inventory = []
            for subgoals in edh_subgoals:
                if subgoals[0] == "Pickup":
                    if holding_item != "Empty":
                        self.logger.debug("Pickup when holding item is not empty!")
                        edh_is_complete = False
                    holding_item = subgoals[1]
                elif subgoals[0] == "Place":
                    if holding_item == "Empty":
                        self.logger.debug("Place when holding item is empty!")
                        edh_is_complete = False
                    holding_item = "Empty"
                subgoals.append(holding_item)
                        
        return edh_is_complete
        
    def synthesize_edh_sessions(self, game_memory: dict):
        """Concatenate dialogue in different edh sessions of a same game instance"""

        dialog_history: List[List[str, str]] = game_memory["dialog_history"]
        future_subgoals: List[List[str, str]] = game_memory["future_subgoals"]
        synthesized_dialog_list = []
        synthesized_dialog_and_subgoals_list = []
        subgoal_counter = 0
        for idx, edh_num in enumerate(game_memory["edh_nums"]):
            edh_dialog = dialog_history[idx]
            edh_subgoals = future_subgoals[idx]
            synthesized_dialog_list += [f"{role}: {text}" for (role, text) in edh_dialog]
            synthesized_dialog_and_subgoals_list += [f"DIALOG {role}: {text}" for (role, text) in edh_dialog]
            synthesized_dialog_and_subgoals_list.append("")
            synthesized_dialog_and_subgoals_list += [f"SUBGOAL {subgoal_counter+k}. {action}({target}), holding[{holding}]" for k, (action, target, holding) in enumerate(edh_subgoals)]
            synthesized_dialog_and_subgoals_list.append("")
            subgoal_counter += len(edh_subgoals)

        synthesized_dialog = "\n".join(synthesized_dialog_list)
        synthesized_dialog_and_subgoals = "\n".join(synthesized_dialog_and_subgoals_list)
        return synthesized_dialog, synthesized_dialog_and_subgoals


    def calc_embeddings(self, task_memory):
        if not self.model:
            self.model = INSTRUCTOR("hkunlp/instructor-xl")
        corpus = []
        for task in task_memory:
            synthesized_dialog, _ = self.synthesize_edh_sessions(task)
            corpus.append(
                ["Represent the household work dialogue", synthesized_dialog]
            )
        self.logger.info("Start calculating embeddings")
        corpus_embeddings = self.model.encode(corpus)
        self.logger.info("Finish calculating embeddings")
        return corpus_embeddings



    def load_memory(self):
        with open(
            f"{self.data_root_dir}/processed_memory/game_memory_{self.memory_split}.json",
            "r",
        ) as f:
            self.task_memory = json.load(f)
        with open(f"{self.data_root_dir}/processed_memory/embeddings_{self.memory_split}.npy", "rb") as f:
            self.memory_embeddings = np.load(f, allow_pickle=True)

        self.logger.info(
            f"{self.memory_split} memory loaded: {len(self.task_memory)} sessions"
        )
        
    def query_task_memory(self, description:str, top_k:int=1):
        if not self.model:
            self.model = INSTRUCTOR("hkunlp/instructor-xl")
        query = [["Represent the household work requirement", description]]
        query_embedding = self.model.encode(query)
        similarities = cosine_similarity(query_embedding, self.memory_embeddings)

        retrieved_task_idxes = similarities.reshape(-1).argsort()[-top_k:][::-1]
        retrieved_tasks = []
        for k, id in enumerate(retrieved_task_idxes):
            self.logger.debug(f"\n================== Closest: {k} ==================\n")
            _, synthesized_dialog_and_subgoals = self.synthesize_edh_sessions(self.task_memory[id])
            self.logger.debug(synthesized_dialog_and_subgoals)
            retrieved_tasks.append(self.task_memory[id])
        return retrieved_tasks