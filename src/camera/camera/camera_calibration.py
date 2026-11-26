import numpy as np
import cv2
import glob

CHESSBOARD_SIZE = (8, 5)
SQUARE_SIZE = 0.039

def main():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

    objpoints = [] 
    imgpoints = []

    image_files = glob.glob("src/camera/resource/*.jpg")
    count=0
    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to load image: {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            count = count + 1
            print(f"processed image {count}")
        else:
            print(f"Chessboard not found in image: {fname}")
    if count == 0:
        print("No valid images found for calibration.")
        return
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera matrix:\n", mtx)
    np.savez("camera_calibration.npz", camera_matrix=mtx, dist_coeff=dist)
    print("saved")
if __name__ == '__main__':
    main()