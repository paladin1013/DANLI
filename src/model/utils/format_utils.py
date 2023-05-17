from typing import List, Dict
from definitions.teach_tasks import GoalArguments, GoalReceptacles, GoalConditions, Operation, OPERATION_EXPLANATION
from difflib import get_close_matches
from model.utils.data_util import process_edh_for_subgoal_prediction
import json

def parse_edh_data(edh_raw, text_dialog_and_act):
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


def match_terms(input_str: str, input_type: str):
    if input_type == "object":
        enums = GoalArguments
    elif input_type == "operation":
        # Some manual alignments
        input_str = input_str.replace("Empty", "Pour").replace("Emptied", "Pour")
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

def load_edh_file(file_path:str):
    with open(f"{file_path}") as f:
        edh_raw = json.load(f)
    edh_text, dialog_history = process_edh_for_subgoal_prediction(edh_raw)
    return edh_raw, edh_text