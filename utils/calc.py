import os
import cv2
import math
import mediapipe as mp
import numpy as np
from scipy.spatial import procrustes

mp_pose = mp.solutions.pose