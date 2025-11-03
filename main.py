import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import math
from ultralytics import YOLO



import pybullet as p
import pybullet_data
import numpy as np


# --- Connect to PyBullet ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0, 0, -9.81)




def spawn_object_by_class(yolo_class, position):
    """
    Spawn a PyBullet object corresponding to a YOLO class at given position.
    position: [x, y, z]
    """
    if yolo_class == "table":
        return p.loadURDF("table/table.urdf", position, globalScaling=np.random.uniform(0,0.2))
    elif yolo_class == "car":
        return p.loadURDF("racecar/racecar.urdf", position, globalScaling=np.random.uniform(0,0.2))
    elif yolo_class == "bottle":
        radius = 0.05
        height = 0.2
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=[0,1,0,1])
        return p.createMultiBody(0, col, vis, position)
    elif yolo_class == "book":
        size = [0.1,0.15,0.02]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[1,0,0,1])
        return p.createMultiBody(0, col, vis, position)
    elif yolo_class == "potted plant":
        # simple cylinder with sphere top
        base_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=0.1)
        base_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=0.1, rgbaColor=[0.55,0.27,0.07,1])
        plant_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.08)
        plant_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[0,1,0,1])
        # base
        base = p.createMultiBody(0, base_col, base_vis, [position[0],position[1],position[2]+0.05])
        # plant top
        plant = p.createMultiBody(0, plant_col, plant_vis, [position[0],position[1],position[2]+0.15])
        return (base, plant)
    else:
        # Default: simple cube
        size = [0.1]*3
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.5,0.5,0.5,1])
        return p.createMultiBody(0, col, vis, position)





# --- Create floor ---
p.loadURDF("plane.urdf")

# --- Load 4-wheeled car ---
car = p.loadURDF("racecar/racecar.urdf", [0,0,0.1])

yolo_detections = ["table", "bottle", "car", "book", "potted plant"]



# --- Random boxes in the environment ---
for i in range(60):
    cls = yolo_detections[np.random.randint(0,5)]
    x, y = np.random.uniform(-2, 2, 2)
    spawn_object_by_class(cls, [x, y, 0])

# --- YOLO model ---
model = YOLO("yolov8s.pt")  # pretrained small YOLOv8

# --- Camera parameters ---
fov, aspect, near, far = 70, 1.0, 0.1, 10.0
img_w, img_h = 128, 128
cam_offset = [0.0, 0.0, 0.5]  # 0.5 m above car

# --- Motion parameters ---
speed = 0.005      # forward movement per step
turn_speed = 0.005 # radians per step
x_min, x_max = -2, 2
y_min, y_max = -2, 2
print("Press 'q' in the OpenCV window to quit.")

def in_bounds(x, y):
    return x_min <= x <= x_max and y_min <= y <= y_max
    
while True:
    # --- Get car position & orientation ---
    pos, orn = p.getBasePositionAndOrientation(car)
    yaw = p.getEulerFromQuaternion(orn)[2]

    # --- Camera setup ---
    cam_pos = [pos[0] + cam_offset[0], pos[1] + cam_offset[1], pos[2] + cam_offset[2]]
    target_pos = [pos[0] + math.cos(yaw), pos[1] + math.sin(yaw), cam_pos[2]]
    view_matrix = p.computeViewMatrix(cam_pos, target_pos, [0,0,1])
    proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # --- Render camera image ---
    img = p.getCameraImage(img_w, img_h, view_matrix, proj_matrix)
    rgb = np.reshape(img[2], (img_h, img_w, 4))[:, :, :3]
    rgb = rgb.astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # --- YOLO object detection ---
    results = model.predict(rgb_bgr, imgsz=64, verbose=False)[0]
    detected_objects = [int(b.cls) for b in results.boxes] if len(results.boxes) > 0 else []
    names = [model.names[cls] for cls in detected_objects] if detected_objects else []
    if names:
        print("Objects ahead:", names)


    
    if names or not in_bounds(pos[0], pos[1]):
        # Turn randomly if object detected or at boundary
        yaw += np.random.choice([-1,1]) * turn_speed

        # Gradually rotate toward center to stay inside boundaries
        center_x, center_y = (x_min + x_max)/2, (y_min + y_max)/2
        angle_to_center = math.atan2(center_y - pos[1], center_x - pos[0])
        yaw_diff = (angle_to_center - yaw + np.pi) % (2*np.pi) - np.pi
        max_turn = turn_speed
        if abs(yaw_diff) > max_turn:
            yaw += max_turn * np.sign(yaw_diff)
        else:
            yaw += yaw_diff
        # Keep position unchanged
        p.resetBasePositionAndOrientation(car, pos, p.getQuaternionFromEuler([0,0,yaw]))
        names = []
    else:
		    # --- Move or turn ---
        dx = speed * math.cos(yaw)
        dy = speed * math.sin(yaw)
        new_x = pos[0] + dx
        new_y = pos[1] + dy
        # Move forward
        new_pos = [new_x, new_y, pos[2]]
        print(new_pos)
        p.resetBasePositionAndOrientation(car, new_pos, p.getQuaternionFromEuler([0,0,yaw]))

  



    # --- Draw bounding boxes ---
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, 
        box)
        cv2.rectangle(rgb_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Car Camera", rgb_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    p.stepSimulation()
    time.sleep(1/240)

cv2.destroyAllWindows()
p.disconnect()
