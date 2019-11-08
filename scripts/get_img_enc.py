#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import pandas as pd


class Converter:
    def __init__(self):
        self.cb = CvBridge()
        self.single_image = None
        self.index = 0
        self.encoder = []
        self.syn_encoder = []
        self.syn_encoder_diff = []
        self.sub_enc = rospy.Subscriber('/odom_enc_enc', Odometry, self.callback_enc)
        self.sub_img = rospy.Subscriber('/front_realsense/color/image_raw', Image, self.callback_img_with_syn_enc)

    def callback_enc(self, message):
        self.encoder = message
        print(self.encoder)

    def callback_img_with_syn_enc(self, message):
        # show and save images
        try:
            self.single_image = self.cb.imgmsg_to_cv2(message, 'passthrough')
        except CvBridgeError as e:
            print(e)

        cv2.imwrite('./single_images' + '/' + str(self.index) + '.jpg', self.single_image)
        self.index += 1

        # save encoder corresponding to current image
        self.syn_encoder.append(self.encoder)
        df = pd.DataFrame([self.encoder, self.syn_encoder[-1]])
        df.to_csv('./csv_for_enc/enc_enc.csv', mode='a')

        # # calculate encoder_diff
        # if len(self.syn_encoder_diff) == 0:
        #     self.syn_encoder_diff.append(0)
        #     df = pd.DataFrame([self.syn_encoder[0], self.syn_encoder_diff[0]],
        #                       index=['syn_encoder', 'syn_encoder_diff'])
        #     df.to_csv('./csv_for_enc/enc_enc.csv')
        # else:
        #     self.syn_encoder_diff.append(self.syn_encoder[-1] - self.syn_encoder[-2])
        #     df = pd.DataFrame([self.syn_encoder[-1], self.syn_encoder_diff[-1]])
        #     df.to_csv('./csv_for_enc/enc_enc.csv', mode='a', header=False)

        # save encoder corresponding to current image as csv file
        # syn_encoder_linear_x.append(encoder_linear.twist.twist.linear.x)
        # syn_encoder_linear_y.append(encoder_linear.twist.twist.linear.y)
        # syn_encoder_linear_z.append(encoder_linear.twist.twist.linear.z)
        # syn_encoder_angular_x.append(encoder_angular.twist.twist.angular.x)
        # syn_encoder_angular_y.append(encoder_angular.twist.twist.angular.y)
        # syn_encoder_angular_z.append(encoder_angular.twist.twist.angular.z)


def main():
    rospy.init_node('get_img_enc')
    get_img_enc = Converter()
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
