from enum import Enum
from pprint import pprint
from typing import Any, Dict, List
from model.model.gpt_api import GPTAPI
import logging
from definitions.teach_tasks import GoalArguments, GoalReceptacles, GoalConditions
from difflib import get_close_matches


class Operation(Enum):
    Cook = 1
    Clean = 2
    PickUp = 3
    FillWithLiquid = 4
    Empty = 5
    Slice = 6
    Boil = 7
    FillWithCoffee = 8
    Open = 9
    Close = 10
    ToggleOn = 11
    ToggleOff = 12


class ActionType(Enum):
    Manipulate = 0
    Place = 1


class LLMSubgoalPredictor:
    def __init__(self, explanation_level:str="brief", ignore_invalid:bool=True):
        self.gpt_api = GPTAPI()
        self.explanation_level = explanation_level
        self.ignore_invalid = ignore_invalid
        self.logger = logging.getLogger("subgoal_predictor")
        
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

    def gen_edh_prompt(self, edh_session: Dict[str, Any]):
        intro = "Suppose you are a household robot and your user will give you some tasks with language instructions. You need to propose some subgoals to complete this goal. Each subgoal can either be a manipulation action or a placing action. For a manipulation action, you need to specify the operation and the object. For a placing action, you need to specify the object and the receptacle. All the possible objects, operations, and receptacles are listed as below. "

        objects: List[str] = edh_session["objects"]
        receptacles: Dict[str, List[str]] = edh_session["receptacles"]
        history: str = edh_session["history"]

        objects_str = "Valid objects: " + ", ".join(objects) + ". "
        operations_str = (
            "Valid operations: " + ", ".join([op.name for op in Operation]) + ". "
        )

        receptacles_str = (
            "Valid receptacles with valid objects in the following bracket: "
        )
        for receptacle, valid_objects in receptacles.items():
            receptacles_str += receptacle + " (" + ", ".join(valid_objects) + "), "
        receptacles_str += ". "

        history_str = f"Please consider the state after following dialogue and action history.\n{history}"

        end_str = f"Please predict a series of subgoals in the format 'Manipulate (operation, objcect)' or 'Place (object, receptacle)' with {self.explanation_level} explanations. Note that you can only hold one object at a time. You have to place the object you are holding before you can pick up another object."

        return f"{intro}\n\n{objects_str}\n\n{operations_str}\n\n{receptacles_str}\n\n{history_str}\n\n{end_str}"

    def gen_formatting_prompt(self):
        return "Please format your predicted subgoals with format 'Manipulate(operation, object)' or 'Place(object, receptacle)' and remove the explanations. object, operation and receptacle can be any one of the valid items above.\nFor example:\n1. Manipulate(PickUp, Knife)\n2. Place(Knife, Sink)"

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


if __name__ == "__main__":
    predictor = LLMSubgoalPredictor()
    # edh_session = edh_file_parser(game_id="24ed9868107a2701_c467", edh_id=0)
    # print(predictor.predict(edh_session))

    test_reply = """1. Place(Potato, CounterTop)
2. Manipulate(PickUp, Knife)
3. Manipulate(PickUp, Potato)
4. Place(Potato, Pot)
5. Manipulate(PickUp, Potato)
6. Place(Potato, Pot)
7. Manipulate(FillWithLiquid, Pot)
8. Place(Pot, StoveBurner)
9. Manipulate(Cook, Pot)
10. Place(Pot, CounterTop)
11. Manipulate(PickUp, Plate)
12. Place(Potatoes, Plate)
"""
    pprint(predictor.parse_gpt_reply(test_reply))
