# Import libraries we need
import cv2                # OpenCV library for working with the webcam and images
import face_recognition   # Library to detect and recognize faces
import os                 # For working with files and directories
import pickle             # For saving and loading Python objects (like face encodings)

# File where we will save all known face encodings (like a small database)
ENCODINGS_FILE = "face_encodings.pkl"

# Load stored face encodings if the file exists, otherwise start with an empty dictionary
if os.path.exists(ENCODINGS_FILE):          # Check if the encodings file already exists
    with open(ENCODINGS_FILE, "rb") as f:   # Open file in read-binary mode
        known_faces = pickle.load(f)        # Load saved data into 'known_faces'
else:
    known_faces = {}  # If no file found, create an empty dictionary {username: [list of encodings]}


# Function to save the 'known_faces' dictionary into the file
def save_encodings():
    with open(ENCODINGS_FILE, "wb") as f:   # Open the file in write-binary mode
        pickle.dump(known_faces, f)         # Save (dump) the dictionary into the file


# Function to add a new user with face encodings
def add_user():
    # Ask the user for a name
    username = input("Enter a username: ").strip()

    # If the username already exists, just add more face samples for them
    if username in known_faces:
        print(f"[WARN] User '{username}' already exists. Adding more samples.")

    # Open the webcam (0 means default camera)
    cap = cv2.VideoCapture(0)
    print(f"[INFO] Adding user: {username}. Press 'q' when done.")
    encodings = []  # A list to store face encodings for this user

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # Convert the frame from BGR (used by OpenCV) to RGB (used by face_recognition)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations in the frame
        # This gives us coordinates (top, right, bottom, left) of each detected face
        boxes = face_recognition.face_locations(rgb)

        # For each detected face in the frame
        for box in boxes:
            # Generate a 128-dimension vector (encoding) that uniquely represents the face
            encoding = face_recognition.face_encodings(rgb, [box])[0]
            
            # Add this encoding to our list of samples
            encodings.append(encoding)

            # Draw a rectangle around the face on the video frame (so we can see it)
            cv2.rectangle(frame, (box[3], box[0]), (box[1], box[2]), (0, 255, 0), 2)

        # Show the video frame in a window
        cv2.imshow("Add User", frame)

        # If user presses 'q', break the loop and stop capturing
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the camera and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

    # If we successfully captured at least one face encoding
    if encodings:
        # If this is a new user, create an entry for them in known_faces
        if username not in known_faces:
            known_faces[username] = []

        # Add all captured encodings for this user
        known_faces[username].extend(encodings)

        # Save updated known_faces into the file
        save_encodings()

        print(f"[INFO] User '{username}' added with {len(encodings)} samples.")
    else:
        print("[WARN] No face detected. User not added.")


# Function to recognize users from webcam in real-time
def recognize_users():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting recognition. Press 'q' to quit.")

    while True:
        # Capture a frame
        ret, frame = cap.read()

        # Convert from BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations in the frame
        boxes = face_recognition.face_locations(rgb)

        # Encode (convert) the faces into 128-dimension vectors
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Loop through every detected face in the frame
        for encoding, box in zip(encodings, boxes):
            name = "Unknown"  # Default label is "Unknown"

            # Compare this encoding with all known faces
            for user, user_encs in known_faces.items():
                # Compare the detected encoding with each known encoding of the user
                matches = face_recognition.compare_faces(user_encs, encoding, tolerance=0.5)

                # If any encoding matches, assign that user's name
                if True in matches:
                    name = user
                    break

            # Draw a rectangle around the face
            cv2.rectangle(frame, (box[3], box[0]), (box[1], box[2]), (0, 255, 0), 2)

            # Put the name above the face rectangle
            cv2.putText(frame, name, (box[3], box[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the live video with annotations
        cv2.imshow("Face Recognition", frame)

        # If user presses 'q', stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()


# Main menu function to interact with the user
def menu():
    while True:
        print("\n===== Facial Recognition System =====")
        print("1. Add User")        # Option to add a new user
        print("2. Recognize Users") # Option to start recognition
        print("3. Exit")            # Exit program
        choice = input("Choose an option (1-3): ").strip()

        if choice == "1":
            add_user()
        elif choice == "2":
            recognize_users()
        elif choice == "3":
            print("[INFO] Exiting program.")
            break
        else:
            print("[ERROR] Invalid choice. Please enter 1-3.")


# Entry point of the program
if __name__ == "__main__":
    menu()
