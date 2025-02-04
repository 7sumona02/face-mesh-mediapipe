import mediapipe as mp
import numpy as np
import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize FaceMesh
facemesh = mp.solutions.face_mesh
face = facemesh.FaceMesh(static_image_mode=True, min_tracking_confidence=0.5, min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils

# Custom drawing specifications
drawing_spec = draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 0, 255)) 

while True:
    # Read a frame from the webcam
    ret, frm = cap.read()
    if not ret:
        break

    # Create a black background image of the same size as the frame
    black_bg = np.zeros_like(frm)

    # Convert the frame to RGB
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

    # Process the frame with FaceMesh
    op = face.process(rgb)

    # Draw the facial landmarks on the black background
    if op.multi_face_landmarks:
        for face_landmarks in op.multi_face_landmarks:
            draw.draw_landmarks(
                image=black_bg,  # Draw on the black background
                landmark_list=face_landmarks,
                connections=facemesh.FACEMESH_TESSELATION,  # Draw the mesh
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    # Display the black background with the facial mesh
    cv2.imshow("window", black_bg)

    # Break the loop if 'ESC' is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()