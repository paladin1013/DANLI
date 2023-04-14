from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import cv2
import matplotlib.pyplot as plt
sam = sam_model_registry["default"](checkpoint="models/sam_vit.pth")
predictor = SamPredictor(sam)
img = cv2.imread("/home/yhgao/repositories/DANLI/teach-dataset/processed_20220610/examples/0a3ed01387e5ec38_4214/commander.frame.38.81817173957825.jpeg")
# plt.savefig(img, "test.jpeg")
masks, _, _ = predictor.predict(point_labels=)
cv2.imwrite("test.jpg", masks[0].astype(np.uint8)*255)