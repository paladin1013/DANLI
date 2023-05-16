from enum import Enum
from pprint import pprint
from typing import Any, Dict, List
from model.model.gpt_api import GPTAPI
from model.data.memory_manager import TaskMemoryManager
import logging
from definitions.teach_tasks import GoalArguments, GoalReceptacles, GoalConditions
from difflib import get_close_matches
import json
from model.utils.data_util import process_edh_for_subgoal_prediction

class Operation(Enum):
    Cook = 1
    Clean = 2
    FillWithLiquid = 4
    Empty = 5
    Slice = 6
    Boil = 7
    FillWithCoffee = 8


class ActionType(Enum):
    Manipulate = 0
    Place = 1


class LLMSubgoalPredictor:
    def __init__(self, explanation_level:str="brief", ignore_invalid:bool=True):
        self.gpt_api = GPTAPI()
        self.explanation_level = explanation_level
        self.ignore_invalid = ignore_invalid
        self.logger = logging.getLogger("subgoal_predictor")
        self.memory_manager = TaskMemoryManager(memory_split="train", data_root_dir="teach-dataset")
        
    def parse_edh_data(self, edh_raw, text_dialog_and_act):
        objects = edh_raw['init_state_diff']['objects']
        
        text_dialog_and_act = text_dialog_and_act
        
        valid_objects:List[str] = []
        valid_receptacles:Dict[str, List[str]] = {}
        for key, value in objects.items():
            name = key.split('|')[0]
            if name not in valid_objects:
                valid_objects.append(name)
            # dist = value['distance']
            if 'receptacleObjectIds' in value.keys():
                recept_names = [recept_str.split('|')[0] for recept_str in value['receptacleObjectIds']]
                # Remove replicated elements in recept_names
                recept_names = list(set(recept_names))
                valid_receptacles.update({name: recept_names})
        edh_session = {}
        edh_session['objects'] = valid_objects
        edh_session['receptacles'] = valid_receptacles
        edh_session['history'] = text_dialog_and_act
        return edh_session

    def gen_edh_prompt(self, edh_session: Dict[str, Any], include_memory=True):
        intro = "Suppose you are a household robot and your user will give you some tasks with language instructions. You need to propose some subgoals to complete this goal. Each subgoal can either be a manipulation action or a placing action. For a manipulation action, you need to specify the operation and the object. For a placing action, you need to specify the object and the receptacle. All the possible objects, operations, and receptacles are listed as below. "

        objects: List[str] = edh_session["objects"]
        receptacles: Dict[str, List[str]] = edh_session["receptacles"]
        history: str = edh_session["history"]

        objects_str = "Valid <object>: \n" + "\n".join(objects) + "\n"
        operations_str = (
            "Valid <operation>: \n" + "\n".join([op.name for op in Operation]) + "\n"
        )

        receptacles_str = (
            "Valid <receptacle> with valid <object> in the following bracket: \n"
        )
        for receptacle, valid_objects in receptacles.items():
            receptacles_str += receptacle + " (" + ", ".join(valid_objects) + ")\n"

        history_str = f"Please consider the state after following dialogue and action history.\n{history}"

        end_str = f"Please predict a series of subgoals in the format 'Manipulate(<operation>, <object>)' or 'Place(<object>, <receptacle>)' for  with {self.explanation_level} explanations. Plese exclude all Manipulate subgoals whose <operation> is not in the operation list."
        
        if not include_memory:
            return f"{intro}\n\n{objects_str}\n\n{operations_str}\n\n{receptacles_str}\n\n{history_str}\n\n{end_str}"
        
        retrieved_tasks = self.memory_manager.query_task_memory(history_str)
        memory_str = "Here are some related examples:\n"
        for task_idx, task in enumerate(retrieved_tasks):
            task_str = f"\n<Example {task_idx+1}>:\nDialog:\n"
            for edh_idx in range(len(task['edh_nums'])):
                for role, sentence in task['dialog_history'][edh_idx]:
                    task_str += f"[{role}]: {sentence}\n"
                        
            task_str += "\nActions:\n"
            cnt = 1
            for edh_idx in range(len(task['edh_nums'])):
                for subj, pred, obj in task['processed_subgoals'][edh_idx]:
                    if pred == "isPickedUp":
                        continue
                    if pred == "parentReceptacles":
                        task_str += f"{cnt}. Place({subj}, {obj})\n"
                    else:
                        pred = pred.replace("simbotIs", "").replace("is", "")
                        operation:Operation = self.match_terms(pred, "operation")
                        task_str += f"{cnt}. Manipulate({operation.name}, {subj})\n"
                    cnt += 1
            
            memory_str += f"{task_str}\n"
            
            return f"{intro}\n\n{objects_str}\n\n{operations_str}\n\n{receptacles_str}\n\n{history_str}\n\n{end_str}\n\n{memory_str}"
            


    def gen_formatting_prompt(self):
        return "Please format your predicted subgoals with format 'Manipulate(<operation>, <object>)' or 'Place(<object>, <receptacle>)' and remove the explanations. <object>, <operation> and <receptacle> can be any one of the valid items above.\nFor example:\n1. Manipulate(PickUp, Knife)\n2. Place(Knife, Sink)"

    def match_terms(self, input_str: str, input_type: str):
        if input_type == "object":
            enums = GoalArguments
        elif input_type == "operation":
            # Some manual alignments
            input_str = input_str.replace("Turn", "Toggle").replace("Operate", "ToggleOn").replace("Rinse", "Clean")
            enums = Operation
        elif input_type == "receptacle":
            enums = GoalReceptacles
        elif input_type == "goal_condition":
            enums = GoalConditions
        else:
            raise (
                ValueError(
                    f"input_type should be one of 'object', 'goal_condition', 'operation', 'receptacle', but got {input_type} instead."
                )
            )
        valid_list = [item.name for item in enums]
        if input_type == "goal_condition":
            valid_list_trimmed = [goal.replace("simbotIs", "").replace("is", "") for goal in valid_list]
            valid = get_close_matches(input_str, valid_list_trimmed, n=1)
            if not valid:
                raise (ValueError(f"{input_str} cannot match a valid {input_type}."))
            return enums(valid_list_trimmed.index(valid[0]))

        valid = get_close_matches(input_str, valid_list, n=1)
        if not valid:
            raise (ValueError(f"{input_str} cannot match a valid {input_type}."))
        return enums[valid[0]]

    def parse_gpt_reply_to_str(self, gpt_reply: str):
        subgoals, parse_error = self.parse_gpt_reply(gpt_reply=gpt_reply)
        # Convert from enum classes to strings
        subgoals_str = []
        for subgoal in subgoals:
            subgoals_str.append([item.name for item in subgoal])
        return subgoals_str, parse_error
        
        
    def parse_gpt_reply(self, gpt_reply: str, output_style: str = "DANLI"):
        """Parse the reply message generated by GPT. The out put will adapt to the structure according to `output_style`
        "DANLI": (pred, subj, obj)
        "new": (Manipulate, object, operation) or (Place, object, receptacle)
        """
        subgoals = []
        idx = 1
        parse_error = False
        for line in gpt_reply.splitlines():
            # Match the format "1. Manipulate(PickUp, Knife)" with ascending idx
            # TODO: add more checks to make sure the format is correct
            try:
                if "Manipulate" in line:
                    try:
                        # In case GPT made the wrong order because of error prompts
                        operation, object = line.split("(")[1].split(")")[0].split(", ")
                        operation = self.match_terms(operation, "operation")
                        object = self.match_terms(object, "object")
                    except ValueError as e:
                        # Swap the orders of object and operation
                        object, operation = line.split("(")[1].split(")")[0].split(", ")
                        operation = self.match_terms(operation, "operation")
                        object = self.match_terms(object, "object")
                            
                    subgoals.append((ActionType.Manipulate, operation, object))

                elif "Place" in line:
                    object, receptacle = line.split("(")[1].split(")")[0].split(", ")
                    object = self.match_terms(object, "object")
                    receptacle = self.match_terms(receptacle, "receptacle")
                    subgoals.append((ActionType.Place, object, receptacle))
            except ValueError as e:
                parse_error = True
                if self.ignore_invalid:
                    self.logger.warning(f"Parsing instruction {line}: {str(e)}")
                    continue
                
        if output_style == "new":
            return subgoals, parse_error
        elif output_style == "DANLI":
            subgoals_DANLI = []
            for subgoal in subgoals:
                try:
                    if subgoal[0] == ActionType.Manipulate:
                        operation = self.match_terms(subgoal[1].name, "goal_condition")
                        object = self.match_terms(subgoal[2].name, "object")
                        subgoals_DANLI.append((operation, object, GoalReceptacles.NONE))
                    elif subgoal[0] == ActionType.Place:
                        object = self.match_terms(subgoal[1].name, "object")
                        receptacle = self.match_terms(subgoal[2].name, "receptacle")
                        subgoals_DANLI.append(
                            (GoalConditions.parentReceptacles, object, receptacle)
                        )
                except ValueError as e:
                    parse_error = True
                    if self.ignore_invalid:
                        self.logger.warning(f"Parsing subgoal for DANLI output {subgoal}: {str(e)}")
                        continue
            subgoals_DANLI.append(
                (GoalConditions.EOS, GoalArguments.NONE, GoalReceptacles.NONE)
            )
            return subgoals_DANLI, parse_error
        else:
            raise NotImplementedError(
                f"output_style {output_style} has not been implemented yet."
            )

    def predict(self, edh_session: Dict[str, Any]):
        replies = self.gpt_api.send(
            [self.gen_edh_prompt(edh_session), self.gen_formatting_prompt()]
        )
        subgoals = self.parse_gpt_reply(replies[1])
        return subgoals
    
    def load_edh_file(self, file_path:str):
        with open(f"{file_path}") as f:
            edh_raw = json.load(f)
        edh_text, dialog_history = process_edh_for_subgoal_prediction(edh_raw)
        return edh_raw, edh_text


    