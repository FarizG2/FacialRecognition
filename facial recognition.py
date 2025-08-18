import cv2
import face_recognition
import os
import pickle

ENCODINGS_FILE = "face_encodings.pkl"

# Load stored encodings (if available)
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}  # {username: [encodings list]}


def save_encodings():
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(known_faces, f)


def add_user():
    username = input("Enter a username: ").strip()
    if username in known_faces:
        print(f"[WARN] User '{username}' already exists. Adding more samples.")

    cap = cv2.VideoCapture(0)
    print(f"[INFO] Adding user: {username}. Press 'q' when done.")
    encodings = []

    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)

        for box in boxes:
            encoding = face_recognition.face_encodings(rgb, [box])[0]
            encodings.append(encoding)
            cv2.rectangle(frame, (box[3], box[0]), (box[1], box[2]), (0, 255, 0), 2)

        cv2.imshow("Add User", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if encodings:
        if username not in known_faces:
            known_faces[username] = []
        known_faces[username].extend(encodings)
        save_encodings()
        print(f"[INFO] User '{username}' added with {len(encodings)} samples.")
    else:
        print("[WARN] No face detected. User not added.")


def recognize_users():
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding, box in zip(encodings, boxes):
            name = "Unknown"

            for user, user_encs in known_faces.items():
                matches = face_recognition.compare_faces(user_encs, encoding, tolerance=0.5)
                if True in matches:
                    name = user
                    break

            cv2.rectangle(frame, (box[3], box[0]), (box[1], box[2]), (0, 255, 0), 2)
            cv2.putText(frame, name, (box[3], box[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def menu():
    while True:
        print("\n===== Facial Recognition System =====")
        print("1. Add User")
        print("2. Recognize Users")
        print("3. Exit")
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


if __name__ == "__main__":
    menu()
