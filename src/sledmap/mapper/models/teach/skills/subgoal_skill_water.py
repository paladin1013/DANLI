from typing import Dict, List, Tuple, Union, Optional
import math, copy
import random

from definitions.teach_object_semantic_class import SemanticClass, check_obj_X_is_Y

from sledmap.mapper.env.teach.teach_subgoal import TeachSubgoal
from sledmap.mapper.models.teach.neural_symbolic_state_repr import NeuralSymbolicAgentState

from sledmap.mapper.env.teach.teach_action import TeachAction
from sledmap.mapper.models.teach.skills.navigation import NavigationSkill
from sledmap.mapper.models.teach.skills.subgoal_skill_get import SubgoalSkillGet
from sledmap.mapper.models.teach.skills.subgoal_skill_place import SubgoalSkillPlace
import sledmap.mapper.models.teach.utils as utils

WaterBottomReceptacles = SemanticClass.get_all_objs_in_semcls("WaterBottomReceptacles")
WaterTaps = SemanticClass.get_all_objs_in_semcls("WaterTaps")
WaterContainers = SemanticClass.get_all_objs_in_semcls("WaterContainers")

class SubgoalSkillWater(NavigationSkill):
    def __init__(
        self,
        logger=None,
    ):
        super().__init__(logger)
        self.get_skill = SubgoalSkillGet(logger=logger)
        self.place_skill = SubgoalSkillPlace(logger=logger)

        # below should appear in reset
        self.STATE = "INIT"  # 
        self.log_func("SubgoalSkillWater is initialized!")
        self.failure_num = 0

    def reset(self):
        super().reset()
        self.get_skill.reset()
        self.place_skill.reset()

        self.failure_num = 0
        self.STATE = "INIT"
        self.log_func("SubgoalSkillWater is reset!")

    def step(
        self,
        subgoal: TeachSubgoal,
        state_repr: NeuralSymbolicAgentState,
    ) -> TeachAction:

        self.num_steps += 1
        for i in range(10):
            if self.num_steps >= self.MAXIMUM_NUM_STEPS:
                self.STATE = "FAILED"
                self.log_func("Failed due to reach step limit")

            if self.STATE == "INIT":
                self.target_type = (
                    subgoal.subject_constraint["semanticClass"]
                    if "semanticClass" in subgoal.subject_constraint
                    else subgoal.subject_constraint["objectType"]
                )
                
                if self.target_type == 'HousePlant' and state_repr.inventory_instance_ids:
                    self.STATE 
                    continue

                self.place_to_sink_sg = TeachSubgoal(
                    predicate="parentReceptacles",
                    subject_constraint={'semanticClass': "WaterContainers"},
                    object_constraint={'semanticClass': "WaterBottomReceptacles"},
                )
                self.STATE = "PLACE_TO_SINK"
                continue

            elif self.STATE == "PLACE_TO_SINK":
                action = self.place_skill.step(self.place_to_sink_sg, state_repr)
                if action.is_stop():
                    self.STATE = "TOGGLE_WATER"
                    continue
                elif action.is_failed():
                    self.STATE = "FAILED"
                    continue
                else:
                    return action
            elif self.STATE == "TOGGLE_WATER":

                if state_repr.last_action.action_type == "ToggleOn":
                    if not state_repr.last_action_failed:
                        if self.target_type != "HousePlant":
                            subgoal.assign_goal_instance_id(self.target_instance.instance_id)
                            self.STATE = "SUCCUSS"
                        else:
                            self.STATE = "GOTO_PLANT"
                        for detection in state_repr.observation.object_detections:
                            if detection.object_type in WaterTaps:
                                target_instance_id = state_repr.get_3D_instance_id_of_detection(detection)
                                return TeachAction.create_action_with_instance(
                                    action_type="ToggleOff",
                                    instance_id_3d=target_instance_id,
                                    detection=detection,
                                )
                        continue
                    else:
                        self.failure_num += 1 
                        if self.failure_num > 5:
                            self.log_func("Too many slice failures!")
                            self.STATE = "FAILED"
                            continue
                        else:
                            self.log_func("Failed to toggle the target. Retry!")
                            action_name = random.choice(["Pan Left", "Pan Right"])
                            return TeachAction(action_name) # break the loop
                
                for detection in state_repr.observation.object_detections:
                    if detection.object_type in WaterTaps:
                        target_instance_id = state_repr.get_3D_instance_id_of_detection(detection)
                        return TeachAction.create_action_with_instance(
                            action_type="ToggleOn",
                            instance_id_3d=target_instance_id,
                            detection=detection,
                        )
                
                self.log_func("Failed to detect the target. Subgoal failed!")
                self.STATE = "FAILED"
                continue
            
            elif self.STATE == "GOTO_PLANT":
                target_instance = self.get_closet_valid_instance(
                    {"objectType": "HousePlant"}, 
                    state_repr, 
                )
                
                if target_instance is not None:
                    action = self.navigate_to(target_instance, state_repr)
                    if action.is_stop():
                        self.log_func("Successfully reached the target!")
                        self.target_instance = target_instance
                        self.STATE = "POUR"
                        continue
                    elif action.is_failed():
                        self.log_func("Failed to reach the target. Subgoal failed!")
                        self.STATE = "FAILED"  # TODO: retry another
                        continue
                    else:
                        return action
                
                else:
                    self.STATE="FALIED"
            
            elif self.STATE == "POUR":
                
                if state_repr.last_action.action_type == "Pour":
                    if not state_repr.last_action_failed:
                        subgoal.assign_goal_instance_id(self.target_instance.instance_id)
                        self.STATE = "SUCCESS"
                        continue
                    else:
                        self.failure_num += 1
                        if self.failure_num >= 5:
                            self.log_func("Failed due to too many failures")
                            self.STATE = "FAILED"
                            continue
                        self.log_func("Failed to toggle the target. Retry!")
                        action_name = random.choice(["Pan Left", "Pan Right"])
                        return TeachAction(action_name) # break the loop
                
                detection = None
                for detection in state_repr.observation.object_detections:
                    if detection.object_type == "HousePlant":
                        break
                target_instance_id = state_repr.get_3D_instance_id_of_detection(detection)
                if detection is not None:
                    return TeachAction.create_action_with_instance(
                        action_type="Pour",
                        instance_id_3d=target_instance_id,
                        detection=detection,
                    )
                else:
                    self.log_func("Failed to detect the target. Subgoal failed!")
                    self.STATE = "FAILED"
                    continue
            
            
            elif self.STATE == "FAILED":
                self.log_func("Subgoal failed after %d steps" % self.num_steps)
                # relax constraint
                self.reset()
                return TeachAction.fail_action()

            elif self.STATE == "SUCCESS":
                self.log_func("Subgoal succeed after %d steps" % self.num_steps)
                self.reset()
                return TeachAction.stop_action()

        self.log_func("Escape from bouncing")
        return TeachAction.fail_action()