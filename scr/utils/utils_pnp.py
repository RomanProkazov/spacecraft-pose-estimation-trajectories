from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2


# Metrics to ecvaluate the pose erros wrt to ground-truth
def error_translation(t_pr, t_gt):
    t_pr = np.reshape(t_pr, (3,))
    t_gt = np.reshape(t_gt, (3,))

    return np.sqrt(np.sum(np.square(t_gt - t_pr)))

def error_orientation(q_pr, q_gt):
    # q must be [qvec, qcos]
    q_pr = np.reshape(q_pr, (4,))
    q_gt = np.reshape(q_gt, (4,))

    qdot = np.abs(np.dot(q_pr, q_gt))
    qdot = np.minimum(qdot, 1.0)
    return np.rad2deg(2*np.arccos(qdot)) # [deg]

def speed_score(t_pr, q_pr, t_gt, q_gt):
    # rotThresh: rotation threshold [deg]
    # posThresh: normalized translation threshold [m/m]
    err_t = error_translation(t_pr, t_gt)
    err_q = error_orientation(q_pr, q_gt) # [deg]

    t_gt = np.reshape(t_gt, (3,))
    speed_t = err_t / np.sqrt(np.sum(np.square(t_gt)))
    speed_q = np.deg2rad(err_q)

    # # Check if within threshold
    # if applyThresh and err_q < rotThresh:
    #     speed_q = 0.0

    # if applyThresh and speed_t < posThresh:
    #     speed_t = 0.0

    speed = speed_t + speed_q

    # # Accuracy of within threshold
    # acc   = float(err_q < rotThresh and speed_t < posThresh)

    return speed

# PnP to solve 3D 2D corrispondeces
def pnp(points_3D, points_2D, cameraMatrix, distCoeffs=None, rvec=None, tvec=None, useExtrinsicGuess=False):
    ''' Perform EPnP using OpenCV.
    Arguments:
        points_3D:  (N,3) numpy.ndarray - 3D coordinates
        points_2D:  (N,2) numpy.ndarray - 2D coordinates
        ...
    Returns:
        q_pr: (4,) numpy.ndarray - unit quaternion (scalar-first)
        t_pr: (3,) numpy.ndarray - position vector (m)
    '''
    if distCoeffs is None:
        distCoeffs = np.zeros((5, 1), dtype=np.float32)

    points_3D = np.ascontiguousarray(points_3D).reshape((-1,1,3))
    points_2D = np.ascontiguousarray(points_2D).reshape((-1,1,2))

    # R_exp - is rotaion in Euler angles
    _, R_exp, t = cv2.solvePnP(points_3D,
                               points_2D,
                               cameraMatrix,
                               distCoeffs, rvec, tvec, useExtrinsicGuess,
                               flags=cv2.SOLVEPNP_EPNP)
    
    # R_pr - Rotation in matrix form
    R_pr, _ = cv2.Rodrigues(R_exp)
    
    # Rotation matrix to quaternion
    q_pr = R.from_matrix(np.squeeze(R_pr)).as_quat()

    # Convert to scalar-first
    q_pr = q_pr[[3,0,1,2]]

    return q_pr, np.squeeze(t)

def quat2dcm(q):
    """ Computing direction cosine matrix from quaternion, adapted from PyNav. 
    Arguments:
        q: (4,) numpy.ndarray - unit quaternion (scalar-first)
    Returns:
        dcm: (3,3) numpy.ndarray - corresponding DCM
    """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm

def project_keypoints(q_vbs2tango, r_Vo2To_vbs, cameraMatrix, distCoeffs, keypoints):
    ''' Project keypoints.
    Arguments:
        q_vbs2tango:  (4,) numpy.ndarray - unit quaternion from VBS to object frame
        r_Vo2To_vbs:  (3,) numpy.ndarray - position vector from VBS to object in VBS frame (m)
        cameraMatrix: (3,3) numpy.ndarray - camera intrinsics matrix
        distCoeffs:   (5,) numpy.ndarray - camera distortion coefficients in OpenCV convention
        keypoints:    (3,N) or (N,3) numpy.ndarray - 3D keypoint locations (m)
    Returns:
        points2D: (2,N) numpy.ndarray - projected points (pix)
    '''
  
    if keypoints.shape[0] != 3:
        keypoints = np.transpose(keypoints)

    # Keypoints into 4 x N homogenous coordinates
    keypoints = np.vstack((keypoints, np.ones((1, keypoints.shape[1]))))

    # transformation to image frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q_vbs2tango)),
                          np.expand_dims(r_Vo2To_vbs, 1)))
    xyz      = np.dot(pose_mat, keypoints) # [3 x N]
    x0, y0   = xyz[0,:] / xyz[2,:], xyz[1,:] / xyz[2,:] # [1 x N] each

    # apply distortion
    r2 = x0*x0 + y0*y0
    cdist = 1 + distCoeffs[0]*r2 + distCoeffs[1]*r2*r2 + distCoeffs[4]*r2*r2*r2
    x  = x0*cdist + distCoeffs[2]*2*x0*y0 + distCoeffs[3]*(r2 + 2*x0*x0)
    y  = y0*cdist + distCoeffs[2]*(r2 + 2*y0*y0) + distCoeffs[3]*2*x0*y0

    # apply camera matrix
    points2D = np.vstack((cameraMatrix[0,0]*x + cameraMatrix[0,2],
                          cameraMatrix[1,1]*y + cameraMatrix[1,2]))

    return points2D


def do_ransac_lm(objectPoints, keypoints, cameraMatrix):
    distCoeffs = np.zeros((5, 1), dtype=np.float32)
    success, R_vec, t_vec, inliers = cv2.solvePnPRansac(objectPoints,
                                                        keypoints,
                                                        cameraMatrix,
                                                        distCoeffs,
                                                        flags=cv2.SOLVEPNP_EPNP,
                                                        reprojectionError=5
                                                        )

    R_vec, t_vec=cv2.solvePnPRefineLM(objectPoints[inliers,:],keypoints[inliers,:],cameraMatrix,distCoeffs,R_vec, t_vec)
    Rotation_matrix, _ = cv2.Rodrigues(R_vec)
    scipy_rotation_matrix=R.from_matrix(Rotation_matrix)
    q_pr=scipy_rotation_matrix.as_quat()
    q_pr = q_pr[[3,0,1,2]]
    
    return  q_pr, t_vec