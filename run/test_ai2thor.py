

# from ai2thor.controller import Controller
# controller = Controller()
# event = controller.step("MoveAhead")
import ai2thor.controller
# from ai2thor.platform import CloudRendering
controller = ai2thor.controller.Controller()
controller.start()

# Kitchens: FloorPlan1 - FloorPlan30
# Living rooms: FloorPlan201 - FloorPlan230
# Bedrooms: FloorPlan301 - FloorPlan330
# Bathrooms: FloorPLan401 - FloorPlan430

print(f"Start Resetting")
controller.reset('FloorPlan28')
print(f"Start Initializing")
controller.step(dict(action='Initialize', gridSize=0.25))

for i in range(10):
    print(f"Start stepping {i}")
    event = controller.step(dict(action='MoveAhead'))
    print(f"Finish stepping {i}")

# Numpy Array - shape (width, height, channels), channels are in RGB order
event.frame

# Numpy Array in BGR order suitable for use with OpenCV
event.cv2image()

# current metadata dictionary that includes the state of the scene
event.metadata
