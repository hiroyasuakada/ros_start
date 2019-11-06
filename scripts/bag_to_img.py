#!/usr/bin/env python

import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

timestamps = []
images = []
for topic, msg, t in rosbag.Bag('/opt/ros/melodic/share/image_view/human_L01_01.bag').read_messages():
    if topic == '/home/ytpc2019b/akada/donkey':
        timestamps.append(t.to_sec())
        images.append(CvBridge().imgmsg_to_cv2(msg, "image_raw"))