# This is a Live IImage capture script tyhat captures images from the webcam,
# organizes it into a folder for each letter and automatically saves them when you press a key

import cv2
import os

dataset_dir = "dataset" # directory for captured images
images_per_letter = 100 # number of images per letter
img_size = 128 # image dimesions

# initialze webcam
cap = cv2.VideoCapture(0) # use webcam (camera index 0)

if not cap.isOpened():
    print("Can't Open the Camera")
    exit()

print("Press SPACE to capture an image, and Q to quit the current letter")

# iterate through each of the letters
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    print(f"Starting image collection for letter: {letter}")
    letter_dir = os.path.join(dataset_dir, letter)
    os.makedirs(letter_dir, exist_ok=True) # creates folder for the letter

    count = 0 
    while count < images_per_letter:
        ret, frame = cap.read() # reads a frame from the webcam
        if not ret:
            print("Can't read frame from the webcam")
            break

        # display the frame
        cv2.putText(frame, f"Letter: {letter}, Count: {count}/{images_per_letter}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Live Image Capture", frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # SPACE to capture an image
            img_path = os.path.join(letter_dir, f"{count}.jpg")
             # Resize the frame to the desired dimensions
            resized_frame = cv2.resize(frame, (img_size, img_size))  # Resize to img_size x img_size
            cv2.imwrite(img_path, resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])  # Save as JPEG

            cv2.imwrite(img_path, resized_frame)  # Save image
            print(f"Captured image: {img_path}")
            count += 1
        elif key == ord('q'):  # Press 'q' to quit the current letter
            break

print("Image collection completed!")
cap.release()
cv2.destroyAllWindows()