#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
#####################  K E I O  U N I V E R S I T Y  ########################
#############################################################################
################################  Python  ###################################
#############################################################################
#############################################################################
#########  Yamaguchi Lab., Department of Administration Engineering  ########
#############################################################################
#############################################################################
###                      donkey_localization.py                           ###
###                                                                       ###
###                   composed by Ayanori Yorozu                          ###
###                   at Keio Advanced Research Centers, Keio University  ###
###                                                                       ###
###                               Copyright (2018), All rights reserved.  ###
###                                                                       ###
###             c/o. Yamagughi Lab., Dept. of Administration Engineering, ###
###             3-14-1 Hiyoshi, Kohoku-ku, Yokohama 223-8522              ###
###             E-mail: yamaguti@ae.keio.ac.jp                            ###
#############################################################################
#############################################################################


################################################################################
#                                   Import                                     #
################################################################################
import	sys
import	rospy
import	copy
import	tf
import	std_msgs
import	sensor_msgs
import	geometry_msgs
import	math
import	tf.transformations
from	std_msgs.msg		import Bool, Float32MultiArray, Float64MultiArray
from	sensor_msgs.msg		import Imu
from	geometry_msgs.msg	import PoseStamped, Point, Quaternion
from	geometry_msgs.msg	import Pose2D, Twist
from	geometry_msgs.msg	import PoseWithCovariance
from	geometry_msgs.msg	import PoseWithCovarianceStamped
from	geometry_msgs.msg	import TransformStamped
from	nav_msgs.msg		import Odometry



################################################################################
#                             Grobal Definition                                #
################################################################################
# Donkey 4 HW --------------------------------------------------------------------
ALPHA		= [0.432, 1.57, -1.57, -0.432]		# 車体幾何中心から各ステア軸への角度 0:LF, 1:LR, 2:RR, 3：RF
D			= 0.7159							# 車体幾何中心から各ステア軸への絶対距離(4号機はLF,RFに対して)
LY			= 0.3								# 機体幾何中心から後輪ステア軸への絶対距離(4号機用)
DD			= [0.1, 0.1, -0.1, -0.1]			# ステア軸から車輪軸中心への距離(4号機はすべて0のため不要)
WEEL_R		= 0.155								# 車輪半径

LF			= 0
LR			= 1
RR			= 2
RF			= 3


# RTK_GNSS param -----------------------------------------------------------------
#BASE_LAT	= 36.461366160						# Base Latitude (deg) 	motegi
#BASE_LON	= 140.171193610						# Base Longitude (deg) 	motegi
#BASE_H		= 146.861000000						# Base H				motegi
BASE_LAT	= 35.99894359						# Base Latitude (deg)
BASE_LON	= 140.27255312						# Base Longitude (deg)
BASE_H		= 31.797							# Base H
EARTH_A 	= 6378137							# semi-major axis (m)
EARTH_E		= 0.081819191 						# eccentricity


# POSE COVARIANCE --------------------------------------------------------------
INITIAL_COV_X		= 0.25
INITIAL_COV_Y		= 0.25
INITIAL_COV_YAW		= 0.1
INITIAL_COV_VX		= 0.25
INITIAL_COV_VY		= 0.25
INITIAL_COV_VYAW	= 0.1


def limitAnglePi(in_angle):
	return (in_angle + math.pi)%(2.0*math.pi) - math.pi


################################################################################
#                            　Localization class                              #
################################################################################
class Localization:
	#===========================================================================
	#   Constructor
	#===========================================================================
	def __init__(self):
		self.p				= Odometry()
		self.trust_flg		= False
		self.p_x_old		= 0.0
		self.p_y_old		= 0.0


	#===========================================================================
	#   Initialize Header, Frame, Pose
	#===========================================================================
	def initialize(self, in_frame_id, in_c_frame_id):
		# header
		self.p.header.stamp		= rospy.Time()
		self.p.header.frame_id	= in_frame_id
		self.p.child_frame_id	= in_c_frame_id

		# pose: PoseWithCovariance
		self.p.pose.pose.position.x	= 0.0
		self.p.pose.pose.position.y	= 0.0
		self.p.pose.pose.position.z	= 0.0
		
		the_p_q	= tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0)
		self.p.pose.pose.orientation.x	= copy.deepcopy(the_p_q[0])
		self.p.pose.pose.orientation.y	= copy.deepcopy(the_p_q[1])
		self.p.pose.pose.orientation.z	= copy.deepcopy(the_p_q[2])
		self.p.pose.pose.orientation.w	= copy.deepcopy(the_p_q[3])

		for i in range(36):
			self.p.pose.covariance[i]	= 0.0
		self.p.pose.covariance[0]		= copy.deepcopy(INITIAL_COV_X)
		self.p.pose.covariance[7]		= copy.deepcopy(INITIAL_COV_Y)
		self.p.pose.covariance[35]		= copy.deepcopy(INITIAL_COV_YAW)

		# velocity: TwistWithCovariance
		self.p.twist.twist.linear.x		= 0.0
		self.p.twist.twist.linear.y		= 0.0
		self.p.twist.twist.linear.z 	= 0.0

		self.p.twist.twist.angular.x	= 0.0
		self.p.twist.twist.angular.y	= 0.0
		self.p.twist.twist.angular.z	= 0.0

		for i in range(36):
			self.p.twist.covariance[i]	= 0.0
		self.p.twist.covariance[0]		= copy.deepcopy(INITIAL_COV_VX)
		self.p.twist.covariance[7]		= copy.deepcopy(INITIAL_COV_VY)
		self.p.twist.covariance[35]		= copy.deepcopy(INITIAL_COV_VYAW)

		# Publisher
		self.pub	= rospy.Publisher(in_frame_id, Odometry, queue_size=10)

		# TF Broadchaster
		self.tf_bc	= tf.TransformBroadcaster()


	#===========================================================================
	#   ResetPose	
	#===========================================================================
	def resetPose(self, in_pose):

		# pose: PoseWithCovariance
		self.p.pose.pose.position.x	= in_pose.pose.pose.position.x
		self.p.pose.pose.position.y	= in_pose.pose.pose.position.y
		self.p.pose.pose.position.z	= in_pose.pose.pose.position.z
		
		self.p.pose.pose.orientation.x	= in_pose.pose.pose.orientation.x
		self.p.pose.pose.orientation.y	= in_pose.pose.pose.orientation.y
		self.p.pose.pose.orientation.z	= in_pose.pose.pose.orientation.z
		self.p.pose.pose.orientation.w	= in_pose.pose.pose.orientation.w

		for i in range(36):
			self.p.pose.covariance[i]	= in_pose.pose.covariance[i]

		# Publisher
		self.odomPub()

		self.tfBc()


	#===========================================================================
	#   Pose Update with ⊿x,⊿y,⊿θ(ENC, IMU)
	#===========================================================================
	def pUpdate(self, in_dp):
		the_orientation = (
			copy.deepcopy(self.p.pose.pose.orientation.x),
			copy.deepcopy(self.p.pose.pose.orientation.y),
			copy.deepcopy(self.p.pose.pose.orientation.z),
			copy.deepcopy(self.p.pose.pose.orientation.w)
			)
		the_rpy	= tf.transformations.euler_from_quaternion(the_orientation)		# ※tuple
		the_roll	= copy.deepcopy(the_rpy[0])
		the_pitch	= copy.deepcopy(the_rpy[1])
		the_yaw		= copy.deepcopy(the_rpy[2])

		# Update position x,y
		self.p.pose.pose.position.x		+= (in_dp.x*math.cos(the_yaw) - in_dp.y*math.sin(the_yaw))
		self.p.pose.pose.position.y		+= (in_dp.x*math.sin(the_yaw) + in_dp.y*math.cos(the_yaw))
	
		# Update yaw
		the_yaw		+= in_dp.theta
		#the_yaw		= limitAnglePi(the_yaw)
		the_q		= tf.transformations.quaternion_from_euler(the_roll, the_pitch, the_yaw)
		self.p.pose.pose.orientation.x	= copy.deepcopy(the_q[0])
		self.p.pose.pose.orientation.y	= copy.deepcopy(the_q[1])
		self.p.pose.pose.orientation.z	= copy.deepcopy(the_q[2])
		self.p.pose.pose.orientation.w	= copy.deepcopy(the_q[3])


	#===========================================================================
	#   Position Update direct x,y(GPS)
	#===========================================================================
	def pUpdateGps(self, in_p):

		# Update position x,y
		self.p.pose.pose.position.x	= copy.deepcopy(in_p.x)
		self.p.pose.pose.position.y	= copy.deepcopy(in_p.y)


	#===========================================================================
	#   Yaw angle Update from GPS move direction
	#===========================================================================
	def yawUpdateGps(self, in_p):
		# Update yaw
		the_yaw		= in_dp.theta
		the_q		= tf.transformations.quaternion_from_euler(the_roll, the_pitch, the_yaw)
		self.p.pose.pose.orientation.x	= copy.deepcopy(the_q[0])
		self.p.pose.pose.orientation.y	= copy.deepcopy(the_q[1])
		self.p.pose.pose.orientation.z	= copy.deepcopy(the_q[2])
		self.p.pose.pose.orientation.w	= copy.deepcopy(the_q[3])


	#===========================================================================
	#   Publish Odometry
	#===========================================================================
	def odomPub(self):
		self.pub.publish(self.p)


	#===========================================================================
	#   Broadcast TF
	#===========================================================================
	def tfBc(self):
		the_trans	= TransformStamped()
		the_trans.header.stamp		= rospy.Time()
		the_trans.header.frame_id	= self.p.header.frame_id
		the_trans.child_frame_id	= self.p.child_frame_id

		the_trans.transform.translation.x	= self.p.pose.pose.position.x
		the_trans.transform.translation.y	= self.p.pose.pose.position.y
		the_trans.transform.translation.z	= self.p.pose.pose.position.z
		the_trans.transform.rotation.x		= self.p.pose.pose.orientation.x
		the_trans.transform.rotation.y		= self.p.pose.pose.orientation.y
		the_trans.transform.rotation.z		= self.p.pose.pose.orientation.z
		the_trans.transform.rotation.w		= self.p.pose.pose.orientation.w

		self.tf_bc.sendTransform(
			(self.p.pose.pose.position.x, self.p.pose.pose.position.y, self.p.pose.pose.position.z),
			(self.p.pose.pose.orientation.x, self.p.pose.pose.orientation.y, self.p.pose.pose.orientation.z, self.p.pose.pose.orientation.w),
			rospy.Time.now(),
			self.p.child_frame_id,
			self.p.header.frame_id
		)



################################################################################
#                            Donkey Localization                               #
################################################################################
class DonkeyLocalization:

	#===========================================================================
	#   Constructor
	#===========================================================================
	def __init__(self):
		print "\n=============== Donkey Localization (Odom + IMU + GNSS) ===================="

		# Initialize node ------------------------------------------------------
		rospy.init_node("donkey_localization")

		# Subscribed data ------------------------------------------------------
		self.v_wheel		= [0.0, 0.0, 0.0, 0.0]			# LF,LR,RR,RF
		self.ang_steer		= [0.0, 0.0, 0.0, 0.0]			# LF,LR,RR,RF
		self.p_enc			= geometry_msgs.msg.Pose2D()
		self.dp_enc			= geometry_msgs.msg.Pose2D()
		self.v_enc			= geometry_msgs.msg.Pose2D()
		self.v_enc_old		= geometry_msgs.msg.Pose2D()
		self.t_enc			= rospy.get_time()

		self.d_rpy_imu		= geometry_msgs.msg.Point()
		self.rpy_imu		= geometry_msgs.msg.Point()
		self.rpy_imu_old	= geometry_msgs.msg.Point()
		self.init_imu_flg	= True
		self.t_imu			= rospy.get_time()

		self.p_gps			= geometry_msgs.msg.PoseWithCovarianceStamped()
		self.t_gps			= rospy.get_time()


		# Estimated pose ------------------------------------------------------
		self.odom_enc_enc	= Localization()
		self.odom_enc_imu	= Localization()
		self.odom_gps_enc	= Localization()
		self.odom_gps_imu	= Localization()
		self.odom_gpsxenc	= Localization()


		# Initialize Poses ---------------------------------------------
		self.odom_enc_enc.initialize('/odom_enc_enc', '/base_footprint_ee')
		self.odom_enc_imu.initialize('/odom_enc_imu', '/base_footprint_ei')
		self.odom_gps_enc.initialize('/odom_gps_enc', '/base_footprint_ge')
		self.odom_gps_imu.initialize('/odom_gps_imu', '/base_footprint_gi')
		self.odom_gpsxenc.initialize('/odom_gpsxenc', '/base_footprint_gex')

		self.p_gps.header.stamp		= rospy.Time()
		self.p_gps.header.frame_id	= '/map'


		# Subscriber --------------------------------------------------
		self.imu_sub		= rospy.Subscriber('/imu/data', Imu, self.subImu)
		self.sensinfo_sub	= rospy.Subscriber('/sensinfo', Float32MultiArray, self.subSensinfo)
		self.gps_sub		= rospy.Subscriber('/donkey/gps', Float64MultiArray, self.subGps)
		self.initialize_sub	= rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.resetPoses)


		# Publisher ----------------------------------------------------
		self.gnss_map_pos_pub		= rospy.Publisher('/donkey/gps_map', PoseWithCovarianceStamped, queue_size=10)

		rospy.spin()


	#===============================================s===========================
	#   Get Current IMU data
	#===========================================================================
	def subImu(self, in_imu):

		the_t		= rospy.get_time()
		the_dt		= the_t - self.t_imu	# dt[s]
		self.t_imu	= copy.deepcopy(the_t)
		the_dp_imu	= geometry_msgs.msg.Pose2D()

		the_orientation = (
			in_imu.orientation.x,
			in_imu.orientation.y,
			in_imu.orientation.z,
			in_imu.orientation.w
		)
		the_rpy	= tf.transformations.euler_from_quaternion(the_orientation)

		self.rpy_imu.x	= the_rpy[0]
		self.rpy_imu.y	= the_rpy[1]
		self.rpy_imu.z	= the_rpy[2]

		if self.init_imu_flg == True:
			self.rpy_imu_old.x	= the_rpy[0]
			self.rpy_imu_old.y	= the_rpy[1]
			self.rpy_imu_old.z	= the_rpy[2]
			self.init_imu_flg	= False

		self.d_rpy_imu.x	= self.rpy_imu.x - self.rpy_imu_old.x
		self.d_rpy_imu.y	= self.rpy_imu.y - self.rpy_imu_old.y
		self.d_rpy_imu.z	= self.rpy_imu.z - self.rpy_imu_old.z
		self.rpy_imu_old	= copy.deepcopy(self.rpy_imu)

		#print self.rpy_imu


		#-------------------------------------------------------------------#
		#							Update each Angle						#
		#-------------------------------------------------------------------#
		the_dp_imu.x		= 0.0
		the_dp_imu.y		= 0.0
		the_dp_imu.theta	= copy.deepcopy(self.d_rpy_imu.z)

		# Enc_Imu
		self.odom_enc_imu.pUpdate(the_dp_imu)

		# Gps_Imu
		self.odom_gps_imu.pUpdate(the_dp_imu)


		#-------------------------------------------------------------------#
		#					Publish and Broadcast each Pose					#
		#-------------------------------------------------------------------#
		# Enc-Imu
		self.odom_enc_imu.odomPub()
		self.odom_enc_imu.tfBc()

		# Gps-Imu
		self.odom_gps_imu.odomPub()
		self.odom_gps_imu.tfBc()


	#===============================================s===========================
	#   Get Current Encoder data
	#===========================================================================
	def subSensinfo(self, in_msg):

		the_t		= rospy.get_time()
		the_dt		= the_t - self.t_enc	# dt[s]
		self.t_enc	= copy.deepcopy(the_t)
		the_dp_enc	= geometry_msgs.msg.Pose2D()


		#-------------------------------------------------------------------#
		#						Get Sensor Information						#
		#-------------------------------------------------------------------#
		# sens_msg: (0:steerLR[rad], 1:steerLF[rad], 2:steerRF[rad], 3:steerRR[rad], 
		#			  4:wheelLR[m/s], 5:wheelLF[m/s], 6:wheelRF[m/s], 7:wheelRR[m/s], 8:t(経過時間[s]))
		# Donkey odom steer/wheel: (LF, LR, RR, RF)
		self.ang_steer[LF]		= in_msg.data[1];
		self.ang_steer[LR]		= in_msg.data[0];
		self.ang_steer[RR]		= in_msg.data[3];
		self.ang_steer[RF]		= in_msg.data[2];
		self.v_wheel[LF]		= in_msg.data[5];
		self.v_wheel[LR]		= in_msg.data[4];
		self.v_wheel[RR]		= in_msg.data[7];
		self.v_wheel[RF]		= in_msg.data[6];


		#-------------------------------------------------------------------#
		#						Calculate enc vx, vy, vth					#
		#-------------------------------------------------------------------#
		# スリップ検知機能追加必要
		#self.v_enc.x			= (self.v_wheel[LR] + self.v_wheel[RR])/2.0;
		#self.v_enc.y			= 0.0;
		#self.v_enc.theta		= (-self.v_wheel[LR] + self.v_wheel[RR])/(2.0*LY);
		self.v_enc.x			= (self.v_wheel[LF] + self.v_wheel[LR] + self.v_wheel[RF] + self.v_wheel[RR])/4.0;
		self.v_enc.y			= 0.0;
		self.v_enc.theta		= (-self.v_wheel[LF] - self.v_wheel[LR] + self.v_wheel[RF] + self.v_wheel[RR])/(4.0*LY);

		self.dp_enc.x			= self.v_enc.x*the_dt
		self.dp_enc.y			= self.v_enc.y*the_dt
		self.dp_enc.theta		= self.v_enc.theta*the_dt


		#-------------------------------------------------------------------#
		#							Update each Pose						#
		#-------------------------------------------------------------------#
		# Enc-Enc
		the_dp_enc.x		= copy.deepcopy(self.dp_enc.x)
		the_dp_enc.y		= copy.deepcopy(self.dp_enc.y)
		the_dp_enc.theta	= copy.deepcopy(self.dp_enc.theta)
		self.odom_enc_enc.pUpdate(the_dp_enc)

		# Enc-Imu
		the_dp_enc.x		= copy.deepcopy(self.dp_enc.x)
		the_dp_enc.y		= copy.deepcopy(self.dp_enc.y)
		the_dp_enc.theta	= 0.0
		self.odom_enc_imu.pUpdate(the_dp_enc)

		# Gps-Enc
		the_dp_enc.x		= 0.0
		the_dp_enc.y		= 0.0
		the_dp_enc.theta	= copy.deepcopy(self.dp_enc.theta)
		self.odom_gps_enc.pUpdate(the_dp_enc)

		# GpsxEnc
		the_dp_enc.x		= copy.deepcopy(self.dp_enc.x)
		the_dp_enc.y		= copy.deepcopy(self.dp_enc.y)
		the_dp_enc.theta	= copy.deepcopy(self.dp_enc.theta)
		self.odom_gpsxenc.pUpdate(the_dp_enc)


		#-------------------------------------------------------------------#
		#					Publish and Broadcast each Pose					#
		#-------------------------------------------------------------------#
		# Enc-Enc
		self.odom_enc_enc.odomPub()
		self.odom_enc_enc.tfBc()

		# Enc-Imu
		self.odom_enc_imu.odomPub()
		self.odom_enc_imu.tfBc()

		# Gps-Enc
		self.odom_gps_enc.odomPub()
		self.odom_gps_enc.tfBc()

		# GpsxEnc
		self.odom_gpsxenc.odomPub()
		self.odom_gpsxenc.tfBc()


	#===============================================s===========================
	#   Get Current Gps data
	#===========================================================================
	def subGps(self, in_gps):

		the_t		= rospy.get_time()
		the_dt		= the_t - self.t_gps	# dt[s]
		self.t_gps	= copy.deepcopy(the_t)
		the_p_gps	= geometry_msgs.msg.Pose2D()


		self.gnss_lat	= copy.deepcopy(in_gps.data[2])
		self.gnss_lon	= copy.deepcopy(in_gps.data[3])
		self.gnss_h		= copy.deepcopy(in_gps.data[4])
		self.gnss_stat	= copy.deepcopy(in_gps.data[5])	# 状態1:Fix, 2:Float, 5:Sigle
		self.gnss_hdop	= copy.deepcopy(in_gps.data[13])

		self.p_gps.pose.pose.position.x	= self.lon2MeterX(self.gnss_lat, self.gnss_lon) 
		self.p_gps.pose.pose.position.y	= self.lat2MeterY(self.gnss_lat)
		self.p_gps.pose.pose.position.z	= 0.0
		for i in range(36):
			self.p_gps.pose.covariance[i]	= 0.0
		self.p_gps.pose.covariance[0]	= in_gps.data[7]**2.0
		self.p_gps.pose.covariance[7]	= in_gps.data[8]**2.0
		self.p_gps.pose.covariance[1]	= self.gnss_hdop**2.0
		self.p_gps.pose.covariance[6]	= self.gnss_hdop**2.0
		self.p_gps.pose.covariance[35]	= INITIAL_COV_YAW


		# Publish
		self.gnss_map_pos_pub.publish(self.p_gps)
		#print self.p_gps




		#-------------------------------------------------------------------#
		#							Update each Pose						#
		#-------------------------------------------------------------------#
		the_p_gps.x		= copy.deepcopy(self.p_gps.pose.pose.position.x)
		the_p_gps.y		= copy.deepcopy(self.p_gps.pose.pose.position.y)
		the_p_gps.theta	= 0.0
		# Gps-Enc
		self.odom_gps_enc.pUpdateGps(the_p_gps)

		# Gps-Imu
		self.odom_gps_imu.pUpdateGps(the_p_gps)

		# GpsxEnc 水平精度1.0m以下，Fix/Float解以上
		#if(self.gnss_hdop < 1.0 and self.gnss_stat < 3):
		#if(self.gnss_hdop < 3.5 and self.gnss_stat < 6):
		self.odom_gpsxenc.pUpdateGps(the_p_gps)

		# 移動量がある程度ある場合は移動方向＝姿勢角：今回は使わない
		#if self.gnss_hdop < 0.25 and self.odom_gpsxenc.trust_flg==True:
		#	the_dist	= math.sqrt((self.odom_gpsxenc.p_x_old-the_p_gps.x)**2.0 + (self.odom_gpsxenc.p_y_old-the_p_gps.y)**2.0)
		#	if the_dist > 0.5 and the_dist < 1.2:
		#		the_p_gps.theta	= math.atan2(self.odom_gpsxenc.p_y_old-the_p_gps.y, self.odom_gpsxenc.p_x_old-the_p_gps.x)
		#		self.odom_gpsxenc.yawUpdateGps(the_p_gps)


		#-------------------------------------------------------------------#
		#					Publish and Broadcast each Pose					#
		#-------------------------------------------------------------------#
		# Gps_Enc
		self.odom_gps_enc.odomPub()
		self.odom_gps_enc.tfBc()

		# Gps-Imu
		self.odom_gps_imu.odomPub()
		self.odom_gps_imu.tfBc()

		# GpsxEnc
		self.odom_gpsxenc.odomPub()
		self.odom_gpsxenc.tfBc()



		if self.gnss_hdop < 0.25:
			self.odom_gpsxenc.trust_flg	= True
			self.odom_gpsxenc.p_x_old	= copy.deepcopy(the_p_gps.x)
			self.odom_gpsxenc.p_y_old	= copy.deepcopy(the_p_gps.y)
		else:
			self.odom_gpsxenc.trust_flg	= False


	#===============================================s===========================
	#   Reset Pose from Rviz
	#===========================================================================
	def resetPoses(self, in_pose):
		self.odom_enc_enc.resetPose(in_pose)
		self.odom_enc_imu.resetPose(in_pose)
		self.odom_gps_enc.resetPose(in_pose)
		self.odom_gps_imu.resetPose(in_pose)
		self.odom_gpsxenc.resetPose(in_pose)



	#===========================================================================
	#   lon2MeterX
	#===========================================================================
	def lon2MeterX(self, in_lat, in_lon):
		the_meter_per_lon	= math.pi/648000.0*(EARTH_A*math.cos(in_lat/180.0*math.pi)) / (math.sqrt(1.0 - EARTH_E*EARTH_E*math.sin(in_lat/180.0*math.pi)*math.sin(in_lat/180.0*math.pi))) * 3600.0
		return (in_lon - BASE_LON)*the_meter_per_lon



	#===========================================================================
	#   lat2MeterY
	#===========================================================================
	def lat2MeterY(self, in_lat):
		the_meter_per_lat	= math.pi/648000.0*(EARTH_A*(1.0 - EARTH_E**2.0)) / math.pow((1.0 - EARTH_E*EARTH_E*math.sin(in_lat/180.0*math.pi)), 1.5) * 3600.0
		return (in_lat - BASE_LAT)*the_meter_per_lat





################################################################################
#                               Main Function                                  #
################################################################################
if __name__ == '__main__':

	localization   = DonkeyLocalization()
