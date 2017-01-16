import sys
import dlib
from skimage import io

predictor_model = "shape_predictor_68_face_landmarks.dat"

# Take the image file name from the command line
inputImage = sys.argv[1]

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

window = dlib.image_window()

# Load the image
image = io.imread(inputImage)

# Run the HOG face detector on the image data
detected_faces = face_detector(image, 1)

print("Found {} faces in the image file {}".format(len(detected_faces), inputImage))

# Show the desktop window with the image
window.set_image(image)

# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):

	# Detected faces are returned as an object with the coordinates
	# of the top, left, right and bottom edges
	print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

	# Draw a box around each face we found
	window.add_overlay(face_rect)

	# Get the the face's pose
	pose_landmarks = face_pose_predictor(image, face_rect)

	# Draw the face landmarks on the screen.
	window.add_overlay(pose_landmarks)

dlib.hit_enter_to_continue()
