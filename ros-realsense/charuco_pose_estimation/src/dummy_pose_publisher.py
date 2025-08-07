#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped

def publish_dummy_poses():
    rospy.init_node('dummy_pose_publisher')
    pub = rospy.Publisher('/mockup_in_cam/pose', PoseStamped, queue_size=10)
    rate = rospy.Rate(10)  # 10Hz
    
    while not rospy.is_shutdown():
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "camera_frame"
        
        # Set some dummy position (1 meter in front of camera)
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 1.0
        
        # Set some dummy orientation (no rotation)
        pose.pose.orientation.w = 1.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        
        pub.publish(pose)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_dummy_poses()
    except rospy.ROSInterruptException:
        pass