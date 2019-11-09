#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import csv
import pandas as pd


class Converter:
    def __init__(self):
        self.cb = CvBridge()
        self.single_image = None
        self.index = 0
        self.encoder = None
        self.syn_encoder = []
        self.syn_encoder_diff = []
        self.sub_enc = rospy.Subscriber('/odom_enc_enc', Odometry, self.callback_enc)
        self.sub_img = rospy.Subscriber('/front_realsense/color/image_raw', Image, self.callback_img_with_syn_enc)

    def callback_enc(self, message):
        self.encoder = message
        print(self.encoder)

    def callback_img_with_syn_enc(self, message):
        # save image
        try:
            self.single_image = self.cb.imgmsg_to_cv2(message, 'passthrough')
        except CvBridgeError as e:
            print(e)

        cv2.imwrite('./single_image' + '/' + str(self.index) + '.jpg', self.single_image)
        self.index += 1

        # save encoder corresponding to current image
        self.syn_encoder.append(self.encoder)

        position_x, position_y, orientation_z, orientation_w = self.value_encoder(self.encoder)
        position_x_diff, position_y_diff, orientation_z_diff, orientation_w_diff \
            = self.diff_value_encoder(self.syn_encoder)

        # header
        # A: 'position_x', B: 'position_y', C: 'orientation_z', D: 'orientation_w',
        #           E: 'position_x_diff', F: 'position_y_diff', G: 'orientation_z_diff', H: 'orientation_w_diff'
        data = [position_x, position_y, orientation_z, orientation_w,
                position_x_diff, position_y_diff, orientation_z_diff, orientation_w_diff]
        with open('./csv_for_enc/enc_enc.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    @staticmethod
    def value_encoder(message):
        position_x = message.pose.pose.position.x
        position_y = message.pose.pose.position.y
        orientation_z = message.pose.pose.orientation.z
        orientation_w = message.pose.pose.orientation.w

        return position_x, position_y, orientation_z, orientation_w

    @staticmethod
    def diff_value_encoder(message_list):
        if len(message_list) == 1:
            position_x_diff = 0
            position_y_diff = 0
            orientation_z_diff = 0
            orientation_w_diff = 0
        else:
            position_x_diff = message_list[-1].pose.pose.position.x - message_list[-2].pose.pose.position.x
            position_y_diff = message_list[-1].pose.pose.position.y - message_list[-2].pose.pose.position.y
            orientation_z_diff = message_list[-1].pose.pose.orientation.z - message_list[-2].pose.pose.orientation.z
            orientation_w_diff = message_list[-1].pose.pose.orientation.w - message_list[-2].pose.pose.orientation.w

        return position_x_diff, position_y_diff, orientation_z_diff, orientation_w_diff


def main():
    rospy.init_node('get_img_enc')
    get_img_enc = Converter()  # call content of init
    rospy.spin()


if __name__ == '__main__':
    main()
