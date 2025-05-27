#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose

class VelManipulator:

    def __init__(self):
        pub_topic_name = "/turtle1/cmd_vel"
        sub_topic_name = "/turtle1/pose"

        self.pub = rospy.Publisher(pub_topic_name, Twist, queue_size=10)
        self.sub = rospy.Subscriber(sub_topic_name, Pose, self.pose_callback)
        self.velocity_msg = Twist()
        
        # State machine variables
        self.state = "moving"  # States: moving, rotating, done
        self.start_theta = 0.0
        self.rotation_threshold = 0.1  # radians tolerance

    def pose_callback(self, msg):
        if self.state == "moving":
            if msg.x >= 7.0:
                # Stop linear movement
                self.velocity_msg.linear.x = 0.0
                # Start rotation
                self.velocity_msg.angular.z = 1.0  # CCW rotation
                self.start_theta = msg.theta
                print(self.start_theta)
                self.state = "rotating"
            else:
                self.velocity_msg.linear.x = 0.5

        elif self.state == "rotating":
            # Calculate normalized angle difference
            delta = msg.theta - self.start_theta
            delta = (delta + math.pi) % (2 * math.pi) - math.pi
            
            if abs(delta) >= math.pi - self.rotation_threshold:
                # Stop rotation
                self.velocity_msg.angular.z = 0.0
                self.state = "moving"

        self.pub.publish(self.velocity_msg)

if __name__ == "__main__":
    rospy.init_node("turtle_motion_controller")
    VelManipulator()
    rospy.spin()