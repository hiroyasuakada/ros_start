#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import csv
import os
import pandas as pd


class Converter:
    def __init__(self):
        self.cb = CvBridge()
        self.single_image = None
        self.index = 0
        self.encoder = None
        self.delta_x_theta = None
        self.syn_encoder = []
        self.syn_encoder_diff = []
        self.sub_enc = rospy.Subscriber('/odom_enc_enc', Odometry, self.callback_enc)
        self.sub_delta_x_theta = rospy.Subscriber('/odom_delta_x_theta', Odometry, self.callback_delta_x_theta)
        self.sub_img = rospy.Subscriber('/front_realsense/color/image_raw', Image, self.callback_img_with_syn_enc)

    def callback_enc(self, message):
        self.encoder = message
        print(self.encoder)

    def callback_delta_x_theta(self, message):
        self.delta_x_theta = message
        print(self.delta_x_theta)

    def callback_img_with_syn_enc(self, message):
        # save image
        try:
            self.single_image = self.cb.imgmsg_to_cv2(message, 'bgr8')
        except CvBridgeError as e:
            print(e)

        self.index += 1
        # cv2.imwrite('./dataset_from_original/new_file' + '/' + str(self.index) + '.jpg', self.single_image)

        # save encoder corresponding to current image
        self.syn_encoder.append(self.encoder)
        position_x, position_y, orientation_z, orientation_w = self.value_encoder(self.encoder)
        position_x_diff, position_y_diff, orientation_z_diff, orientation_w_diff \
            = self.diff_value_encoder(self.syn_encoder, self.encoder)

        delta_x, theta, _, _ = self.value_encoder(self.delta_x_theta)

        # header
        # A: 'delta_x', B: 'theta', C: 'orientation_z', D: 'orientation_w',
        #           E: 'position_x_diff', F: 'position_y_diff', G: 'orientation_z_diff', H: 'orientation_w_diff'
        data = [self.index, position_x, position_y, orientation_z, orientation_w,
                position_x_diff, position_y_diff, orientation_z_diff, orientation_w_diff]
        with open('./dataset_from_original/delta_x_theta.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    @staticmethod
    def value_encoder(message):
        if message is not None:
            position_x = message.pose.pose.position.x
            position_y = message.pose.pose.position.y
            orientation_z = message.pose.pose.orientation.z
            orientation_w = message.pose.pose.orientation.w
        else:
            position_x = ' - '
            position_y = ' - '
            orientation_z = ' - '
            orientation_w = ' - '

        return position_x, position_y, orientation_z, orientation_w

    @staticmethod
    def diff_value_encoder(message_list, message):
        if message is not None:
            if len(message_list) == 1:
                position_x_diff = 0
                position_y_diff = 0
                orientation_z_diff = 0
                orientation_w_diff = 0

            elif message_list[-1] and message_list[-2] is not None:
                position_x_diff = message_list[-1].pose.pose.position.x - message_list[-2].pose.pose.position.x
                position_y_diff = message_list[-1].pose.pose.position.y - message_list[-2].pose.pose.position.y
                orientation_z_diff = message_list[-1].pose.pose.orientation.z - message_list[-2].pose.pose.orientation.z
                orientation_w_diff = message_list[-1].pose.pose.orientation.w - message_list[-2].pose.pose.orientation.w

            else:
                position_x_diff = ' - '
                position_y_diff = ' - '
                orientation_z_diff = ' - '
                orientation_w_diff = ' - '

        else:
            position_x_diff = ' - '
            position_y_diff = ' - '
            orientation_z_diff = ' - '
            orientation_w_diff = ' - '

        return position_x_diff, position_y_diff, orientation_z_diff, orientation_w_diff


def main():
    # os.mkdir('./dataset_from_original/new_file')
    rospy.init_node('get_img_enc')
    get_img_enc = Converter()  # call content of init
    rospy.spin()


if __name__ == '__main__':
    main()
