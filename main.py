import cv2
import os
from datetime import datetime

from face_system import FaceRecognitionSystem
from database import AttendanceDatabase


class AttendanceSystem:

    def __init__(self, students_folder="students"):
        self.students_folder = students_folder
        self.face_system = FaceRecognitionSystem(students_folder)
        self.db = AttendanceDatabase("attendance_records")

        self.cap = None
        self.running = True

        self.frame_count = 0
        self.recognized_faces = {}

        self.faces_detected = 0
        self.recognized_faces_count = 0
        self.unknown_faces = 0

    def initialize_camera(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def mark_attendance(self, student_id, confidence):

        current_time = datetime.now()

        student_name = self.face_system.students.get(
            student_id,
            {}
        ).get("name")

        if not student_name:
            return False

        if student_id in self.recognized_faces:

            last = self.recognized_faces[student_id]

            if (current_time - last).seconds < 300:
                return False

        self.recognized_faces[student_id] = current_time

        self.db.mark_attendance(
            student_name,
            current_time
        )

        print(f"✅ Attendance Marked : {student_name}")

        return True

    def run(self):

        if self.cap is None:
            self.initialize_camera()

        print("🎥 Attendance Started")

        while self.running:

            ret, frame = self.cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            self.frame_count += 1

            results = []

            if self.frame_count % 5 == 0:

                small = cv2.resize(
                    frame,
                    (0, 0),
                    fx=0.25,
                    fy=0.25
                )

                results = self.face_system.recognize_faces(
                    small
                )

            self.faces_detected = len(results)

            recognized = 0
            unknown = 0

            for result in results:

                x, y, w, h = result["bbox"]

                x *= 4
                y *= 4
                w *= 4
                h *= 4

                sid = result["student_id"]
                conf = result["confidence"]

                if sid == "Unknown":
                    unknown += 1
                    color = (0, 0, 255)
                    label = "Unknown"

                else:

                    recognized += 1

                    name = self.face_system.students[sid]["name"]

                    color = (0, 255, 0)
                    label = f"{name} ({conf:.2f})"

                    if conf < 0.55:
                        self.mark_attendance(sid, conf)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            self.recognized_faces_count = recognized
            self.unknown_faces = unknown

            cv2.putText(
                frame,
                f"Present : {len(self.recognized_faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("Attendance System", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                self.running = False

            elif key == ord("s"):
                self.db.save_attendance()
                print("💾 Attendance Saved")

        self.cleanup()

    def stop(self):
        self.running = False

    def cleanup(self):
        """Cleanup resources"""

        self.db.save_attendance()

        if self.cap is not None:
            self.cap.release()

        cv2.destroyAllWindows()
        print("👋 System stopped")


def start_attendance():

    system = AttendanceSystem()
    os.makedirs("students", exist_ok=True)
    os.makedirs("attendance_records", exist_ok=True)

    system.run()
if __name__ == "__main__":

    start_attendance()
    