from model.model.llm_subgoal_predictor import LLMSubgoalPredictor


predictor = LLMSubgoalPredictor()
# edh_session = edh_file_parser(game_id="24ed9868107a2701_c467", edh_id=0)
edh_file_path = "teach-dataset/edh_instances/valid_unseen/0b42b1e6a5ad92ee_8867.edh4.json"
edh_raw, edh_text = predictor.load_edh_file(edh_file_path)
edh_input = predictor.parse_edh_data(edh_raw, edh_text["text_dialog_and_act"])
print(predictor.gen_edh_prompt(edh_input))