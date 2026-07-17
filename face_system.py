import face_recognition
import cv2
import numpy as np
import os
import pickle
from typing import List, Dict


class FaceRecognitionSystem:
    def __init__(self, students_folder: str):
        self.students_folder = students_folder

        # 🔥 Use dictionary (BEST approach)
        # { student_id: { "name": ..., "encoding": ... } }
        self.students = {}

        self.load_students()

    # ================= LOAD STUDENTS =================
    def load_students(self):
        """Load students from images + saved encodings"""
        self.students = {}

        # -------- Load from images --------
        if os.path.exists(self.students_folder):
            for filename in os.listdir(self.students_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):

                    path = os.path.join(self.students_folder, filename)

                    # Extract ID and name
                    name_parts = os.path.splitext(filename)[0].split('_', 1)

                    if len(name_parts) >= 2:
                        student_id = name_parts[0]
                        student_name = name_parts[1].replace('_', ' ')
                    else:
                        student_id = name_parts[0]
                        student_name = student_id

                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        if student_id not in self.students:   # 🔥 prevents overwrite
                         self.students[student_id] = {
                         "name": student_name,
                         "encoding": encodings[0]
        }
                        print(f"✅ Loaded: {student_name} (ID: {student_id})")
                    else:
                        print(f"⚠️ No face found in {filename}")

      

    # ================= FACE RECOGNITION =================
    def recognize_faces(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Faster detection
        face_locations = face_recognition.face_locations(
        rgb_frame,
        model="hog"
    )

        face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )

        results = []

    # No students
        if len(self.students) == 0:
            return results

        known_encodings = []
        student_ids = []

        for student_id, data in self.students.items():
            known_encodings.append(data["encoding"])
            student_ids.append(student_id)

    # Process each detected face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            matches = face_recognition.compare_faces(
            known_encodings,
            face_encoding,
            tolerance=0.6
        )

            distances = face_recognition.face_distance(
            known_encodings,
            face_encoding
        )

            if len(distances) == 0:
             continue

            sorted_distances = np.sort(distances)

            if len(sorted_distances) > 1:
                if abs(sorted_distances[0] - sorted_distances[1]) < 0.03:
                    print("⚠️ Similar match detected")
                    continue

            best_match_index = np.argmin(distances)

            print(
            "Student:",
            student_ids[best_match_index],
            "Distance:",
            distances[best_match_index]
             )

            best_distance = distances[best_match_index]
    
            if best_distance < 0.45:
                student_id = student_ids[best_match_index]
                confidence = best_distance
            else:
                student_id = "Unknown"
                confidence = best_distance
                
            results.append({
            "bbox": (left, top, right - left, bottom - top),
            "student_id": student_id,
            "confidence": confidence
        })

        return results

    # ================= REGISTER STUDENT =================
    def register_student(self, student_id, student_name, video_capture):
        print(f"\n🆕 Registering {student_name} (ID: {student_id})")

        while True:
            ret, frame = video_capture.read()
            frame = cv2.flip(frame, 1)

            # 🔥 Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            if not ret:
                continue

            rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # 🔥 Faster HOG model
            faces = face_recognition.face_locations(rgb, model="hog")

            cv2.imshow("Register - Press SPACE", frame)

            key = cv2.waitKey(1)

            if key == 32:  # SPACE
                if len(faces) != 1:
                    print("❌ Ensure exactly ONE face is visible")
                    continue
                # Scale face back
                
                encoding = face_recognition.face_encodings(rgb, faces)[0]

                # Save encoding
                self.save_encoding(student_id, encoding)

                # Save image
                os.makedirs("students", exist_ok=True)

                filename = f"{student_id}_{student_name.replace(' ', '_')}.jpg"

                cv2.imwrite(
                os.path.join("students", filename),
                frame
                )

                print(f"✅ {student_name} registered successfully!")

                # Reload students instantly
                self.load_students()

                break

            elif key == 27:  # ESC
                break

        cv2.destroyWindow("Register - Press SPACE")

    # ================= SAVE ENCODING =================
    def save_encoding(self, student_id, encoding):
        file = "encodings.pkl"

        if os.path.exists(file):
            with open(file, "rb") as f:
                data = pickle.load(f)
        else:
            data = {"ids": [], "encodings": []}

        if student_id in data["ids"]:
            print("⚠️ Student already exists!")
            return

        data["ids"].append(student_id)
        data["encodings"].append(encoding)

        with open(file, "wb") as f:
            pickle.dump(data, f)