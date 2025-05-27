# If not working, first do: sudo rm -rf /tmp/.docker.xauth
# It still not working, try running the script as root.
## Build the image first
### docker build -t ros1_rs_custom .
## then run this script
xhost local:root


XAUTH=/tmp/.docker.xauth


docker run -it \
    --name=ros1_rs_custom2 \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/home/roman/catkin_ws:/catkin_ws" \
    --volume="/home/roman/spacecraft-pose-estimation-trajectories:/spacecraft-pose-estimation-trajectories" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --net=host \
    --privileged \
    ros_rs_image:latest bash   # osrf/ros:noetic-desktop 

echo "Done."
