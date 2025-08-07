#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool
from charuco_pose_estimation.srv import TestResponse, TestResponseResponse

class TestResponseService:
    def __init__(self):
        rospy.init_node('test_response_service')
        
        # Service
        self.service = rospy.Service(
            'test_response_service', 
            TestResponse, 
            self.handle_request
        )
        
        rospy.loginfo("Test Response Service Ready")

    def handle_request(self, req):
        """Service handler that always returns successful test response"""
        try:
            response = TestResponseResponse()
            response.success = True
            response.message = "Service test successful"
            response.test_data = "Sample test data payload"
            return response
            
        except Exception as e:
            rospy.logerr(f"Service error: {str(e)}")
            response = TestResponseResponse()
            response.success = False
            response.message = str(e)
            response.test_data = ""
            return response

if __name__ == "__main__":
    try:
        TestResponseService()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass