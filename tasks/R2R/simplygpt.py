import json
import os
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from env import R2RBatch
from utils import load_datasets, load_nav_graphs
from agent import BaseAgent, StopAgent, RandomAgent, ShortestAgent

import MatterSim
import time
import math
import cv2
import re
from PIL import Image
from llm_utils.gpt import get_gpt_response, get_gpt_angle, get_history_sum
from llm_utils.blip import get_blip_response

WIDTH = 800
HEIGHT = 600
VFOV = math.radians(60)
HFOV = VFOV*WIDTH/HEIGHT
TEXT_COLOR = [230, 40, 40]

# cv2.namedWindow('Python RGB')
# cv2.namedWindow('Python Depth')

sim = MatterSim.Simulator()
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(VFOV)
sim.setDepthEnabled(False) # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
sim.initialize()
#sim.newEpisode(['2t7WUuJeko7'], ['1e6b606b44df4a6086c0f97e826d4d15'], [0], [0])
#sim.newEpisode(['1LXtFkjw3qL'], ['0b22fa63d0f54a529c525afbf2e8bb25'], [0], [0])
# sim.newRandomEpisode(['1LXtFkjw3qL'])

heading = 0
elevation = 0
location = 0
ANGLEDELTA = 5 * math.pi / 180

env = R2RBatch(None, batch_size=1, splits=['val_unseen'])
env.reset_epoch()
obs = env.reset()


scan = obs[0]['scan']
viewpoint = obs[0]['viewpoint']
heading = obs[0]['heading']
elevation = obs[0]['elevation']

sim.newEpisode([scan], [viewpoint], [heading], [elevation])

def nextline(sentence):
    senlist = re.split(',|\.', sentence)
    senlist = [i + '\r\n' for i in senlist]
    goalsen = ''
    for s in senlist:
        goalsen += s
    return goalsen

tralog = []
progressrate = ""

while True:
    location = 0
    heading = 0
    elevation = 0

    teacher = obs[0]['teacher']
    instructions = obs[0]['instructions']

    state = sim.getState()[0]
    locations = state.navigableLocations
    rgb = np.array(state.rgb, copy=True)
    irgb = np.array(state.rgb, copy=True)

    # if state.location.viewpointId == obs[0][]

    navchoices = {}

    for idx, loc in enumerate(locations[1:]):
        # Draw actions on the screen
        fontScale = 3.0/loc.rel_distance
        x = int(WIDTH/2 + loc.rel_heading/HFOV*WIDTH)
        y = int(HEIGHT/2 - loc.rel_elevation/VFOV*HEIGHT)
        cv2.putText(irgb, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale, TEXT_COLOR, thickness=3)
        
        navchoices[idx + 1] = {'heading_degree': loc.rel_heading*180/math.pi}

    senlist = re.split(',|\.', instructions)

    # Draw instructions on the screen
    x = 10
    y = 10
    y_offset = 0
    for line in senlist:
        (line_width, line_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness=1)
        y += line_height + y_offset
        cv2.putText(irgb, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=1)
        y_offset = int(0.3 * line_height)

    cv2.imwrite('/root/Matterport3DSimulator/output/images/irgb.jpg', irgb)
    cv2.imwrite('/root/Matterport3DSimulator/output/images/rgb.jpg', rgb)

    # use gpt to decide the next step
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    raw_img = Image.fromarray(rgb).convert("RGB")
    raw_img.save('/root/Matterport3DSimulator/output/images/pilrgb.jpg', 'JPEG')

    blip_rps = get_blip_response(raw_img)

    if progressrate is None:
        progressrate = instructions
    
    print('\n------------------------------- Current View -------------------------------\n', blip_rps, '\n')

    if len(locations) > 1: # if any choice is in the view
        gpt_rps = get_gpt_response(progressrate, blip_rps, navchoices)

        print('------------------------------- Reason -------------------------------\n', gpt_rps, '\n')

        infos = gpt_rps.split('#') # '#' is the split feature

        location = int(infos[0])
        heading = float(infos[1]) * math.pi / 180
        reason = infos[2]

        sim.makeAction([location], [heading], [elevation])
    else:
        print('\n------------------------------- Reason -------------------------------\n', gpt_rps, '\n')

        gpt_rps = get_gpt_angle(progressrate, blip_rps)
        infos = gpt_rps.split('#')
        heading = float(infos[0]) * math.pi / 180
        reason = infos[1]

        sim.makeAction([0], [heading], [elevation])
    
    tralog.append({'scene': blip_rps, 'decision_reason': reason})

    progressrate = get_history_sum(instructions, tralog)
    print('\n------------------------------- Degree of progress -------------------------------\n', progressrate, '\n')






