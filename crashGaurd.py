import torch.nn.functional as F
import threading
import time
import os
import torch
import torchvision
import cv2
import numpy as np
import traitlets
import ipywidgets.widgets as widgets
from jupyter_clickable_image_widget import ClickableImageWidget
from IPython.display import display
from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg
from evdev import InputDevice, categorize, ecodes
from jetracer.nvidia_racecar import NvidiaRacecar

# Global variables
model = None
torch_device = None
normalize = None

block = 0
ai_run = False
terminate = False
was_moving = True
threshold = 0.5
off_count = 0
off_time = 0
ai_sleep_time = 0.05
steering_offset = 0.1


# ML methods
def preprocess(camera_value):
    global torch_device, normalize
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(torch_device)
    x = x[None, ...]
    return x

def loadModelIntoGpu():
    global model, torch_device, normalize
    model = torchvision.models.alexnet(pretrained=False)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    model.load_state_dict(torch.load('/home/jetson/blockModel.pth'))
    torch_device = torch.device('cuda')
    model = model.to(torch_device)
    mean = 255.0 * np.array([0.485, 0.456, 0.406])
    stdev = 255.0 * np.array([0.229, 0.224, 0.225])
    normalize = torchvision.transforms.Normalize(mean, stdev)
    print("Model loaded into GPU")
        
def setUpModel():
    model_thread = threading.Thread(target=loadModelIntoGpu, args=())
    model_thread.daemon = True
    model_thread.start()

def ai_loop():
    global model, camera, block, car, ai_run, terminate, threshold, was_moving, ai_sleep_time
    while not terminate:
        if ai_run:
            x = preprocess(camera.value)
            y = model(x)
            y = F.softmax(y, dim=1)
            prob_blocked = float(y.flatten()[0])
            block = prob_blocked
            if prob_blocked > threshold:
                if was_moving:
                    was_moving = False
                    car.throttle = 1
                    time.sleep(0.25)
                    car.throttle = 0
            else:
                was_moving = True
        else:
            was_moving = True
            block = 0
        time.sleep(ai_sleep_time)
        
def startAI():
    ai_thread = threading.Thread(target=ai_loop, args=())
    ai_thread.daemon = True
    ai_thread.start()
    print("AI running")

# Car control methods
def getJoyStick():
    device = "/dev/input/by-id/usb-ShanWan_PC_PS3_Android-event-joystick"
    while not os.path.exists(device):
        time.sleep(2)
    print("Joystick found")
    return InputDevice(device)

def setUpCar():
    global steering_offset
    car = NvidiaRacecar()
    car.throttle_gain = 1
    car.steering_offset=0
    car.steering = steering_offset
    print("Car ready")
    return car

def setUpCamera():
    camera = CSICamera(width=224, height=224, capture_fps=20)
    camera.running = True
    print("Camera up")
    return camera

def wiggle():
    global car
    car.steering = 1
    time.sleep(0.5)
    car.steering = -1
    time.sleep(0.5)
    car.steering = 0


def controlLoop():
    global joystick, block, threshold, car, ai_run, steering_offset
    for event in joystick.read_loop():
        # Analog
        if event.type == ecodes.EV_ABS:
            absevent = categorize(event)
            #print(str(ecodes.bytype[absevent.event.type][absevent.event.code]) + " : " + str(absevent.event.value))
            stick = ecodes.bytype[absevent.event.type][absevent.event.code]
            value = absevent.event.value

            if stick == "ABS_Y":
                if value < 127 and block > threshold:
                    if car.throttle < 0:
                        car.throttle = 0
                    continue
                car.throttle = (value - 127.0)/127.0
            elif stick == "ABS_Z":
                car.steering = ((value - 127.0)*-1/127.0) + steering_offset 

        # Buttons
        elif event.type == ecodes.EV_KEY:
            #print(event.code)
            #print(event.value)
            if event.code == 310:
                # L1
                if block < threshold or event.value == 0:
                    car.throttle = event.value * -0.7
            elif event.code == 312:
                # L2
                if block < threshold or event.value == 0:
                    car.throttle  = event.value * -1
            elif event.code == 311:
                # R1
                car.throttle = event.value * 0.7
            elif event.code == 313:
                # R2
                car.throttle = event.value * 1
            elif event.code == 315 and event.value == 1:
                # Start
                if off_time == 0 or off_time - time.time() > 3:
                    off_count = 1
                    off_time = time.time()
                elif off_count == 1:
                    off_count = 2
                elif off_count == 2:
                    os.system("sudo poweroff")
            elif event.code == 314 and event.value == 1:
                # Select 
                if ai_run:
                    ai_run = False
                else:
                    ai_run = True
                    wiggle()

print("Init Start")
joystick = getJoyStick()
setUpModel()
car = setUpCar()
camera = setUpCamera()
startAI()
wiggle()
print("Init Complete")
controlLoop()
terminate = True
