import cv2 as cv
from cv2 import aruco
import numpy as np

# declare the variable
MARKER_SIZE = 0.1 #pixel 
dist_coef = np.zeros((4,1))
calib_data_path="./calib_data/MultiMatrix.npz"
calib_data=np.load(calib_data_path)
cam_mat=calib_data["camMatrix"]
r_vectors=calib_data["rVector"]
t_vectors=calib_data["tVector"]
# declare the type of aruco dictionary
marker_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

param_markers = aruco.DetectorParameters_create()
# Define the source and destination points for the affine transform
#(điều chỉnh tọa độ gốc và tọa độ của màn hình chiếu ở đây)


# get the camera
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    src_points = np.float32([[0, 0], [[frame.shape[1]-1, 0], 0], [0, frame.shape[0]-1]])
    dst_points = np.float32([[0, 0], [int(0.6*(frame.shape[1]-1)), 0], [int(0.4*(frame.shape[1]-1)), frame.shape[0]-1]])
    cam_mat[0, 2] = 0  # set principal point x-coordinate to 0
    cam_mat[1, 2] = 0
    cam_mat[1,2] = -abs(cam_mat[1, 2])
    if not ret:
        break
    # transfer the color of image to gray     
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, _ = aruco.detectMarkers(gray_frame, marker_dict, parameters=param_markers)
    
     # Get the affine transform matrix
    M = cv.getAffineTransform(src_points, dst_points)
    
    # Apply the affine transform to the frame
    m_frame = cv.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    if marker_corners:
    # estimate posing
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(marker_corners, MARKER_SIZE, cam_mat, dist_coef) 
        for ids, corners, rvec, tvec in zip(marker_IDs, marker_corners, rvecs, tvecs):
            cv.polylines(frame, [corners.astype(np.int32)], True, (0, 255 , 0), 1, cv.LINE_AA)
            # Apply the affine transform to the corners
            corners = np.array(corners, dtype=np.float32)
            corners = cv.transform(corners[None, :, :], M)[0, :, :]
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()
            # Draw the marker and its ID
            cv.polylines(m_frame, [corners], True, (0, 255, 0), 2)
            cv.putText(
                m_frame,
                f"ID: {ids[0]} - Position: ({round(tvec[0][0], 2)}, {round(tvec[0][1], 2)}, {round(tvec[0][2], 2)})",
                bottom_left,
                cv.FONT_HERSHEY_PLAIN,
                1.0,
                (200, 100, 255),
                2,
                cv.LINE_AA,
            )
            print(f"ID: {ids[0]} - Position: ({round(tvec[0][0], 2)}, {round(tvec[0][1], 2)}, {round(tvec[0][2], 2)})")
        # show all of camera
    cv.imshow("CamInital",frame)
    cv.imshow("CAMERA", m_frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
