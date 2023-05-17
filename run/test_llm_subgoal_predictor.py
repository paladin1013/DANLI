from model.model.llm_subgoal_predictor import LLMSubgoalPredictor
import json
from model.utils.format_utils import parse_edh_data, load_edh_file
predictor = LLMSubgoalPredictor()
# edh_session = edh_file_parser(game_id="24ed9868107a2701_c467", edh_id=0)
game_id = "345624f0e74dc947_8525"
edh_num = 5

edh_file_path = f"teach-dataset/edh_instances/valid_unseen/{game_id}.edh{edh_num}.json"
edh_raw, edh_text = load_edh_file(edh_file_path)
edh_input = parse_edh_data(edh_raw, edh_text["text_dialog_and_act"])
example_num = 3
prompt = predictor.gen_edh_prompt(edh_input, example_num=example_num)
print(prompt)
predictor.save_manual_response(f"{game_id}-edh{edh_num}-example_num{example_num}", prompt)
reply = predictor.load_manual_response(f"{game_id}-edh{edh_num}-example_num{example_num}")
print("\nParsed result:\n", predictor.parse_gpt_reply_to_str(reply))


with open("teach-dataset/processed_20220610/games_gt_subgoals.json") as f:
    gt_subgoals_all = json.load(f)
gt_subgoals_sample = gt_subgoals_all[game_id]
idx = gt_subgoals_sample["edh_to_gt_subgoal_idx"][f"edh{edh_num}"]
gt_subgoals = [sg for k, sg in enumerate(gt_subgoals_sample["gt_all_subgoals"]) if k in idx]
print("\nGroundtruth subgoals:\n", gt_subgoals)