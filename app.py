from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os
import cv2
import threading
from datetime import datetime
from flask import session, send_from_directory
from face_system import FaceRecognitionSystem
from main import start_attendance, AttendanceSystem
app = Flask(__name__)
app.secret_key = "attendance_secret_key"
attendance_system = None

# Initialize face system
face_system = FaceRecognitionSystem("students")

# Folders
ATTENDANCE_FOLDER = "attendance_records"
STUDENTS_FOLDER = "students"

os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)
os.makedirs(STUDENTS_FOLDER, exist_ok=True)

@app.route("/camera_stats")
def camera_stats():

    global attendance_system

    if attendance_system:

        return {
            "faces": attendance_system.faces_detected,
            "recognized": attendance_system.recognized_faces_count,
            "unknown": attendance_system.unknown_faces
        }

@app.route("/reports")
def reports():

    total_students = len(face_system.students)

    files = sorted(
        [f for f in os.listdir(ATTENDANCE_FOLDER) if f.endswith(".csv")],
        reverse=True
    )

    selected_file = request.args.get("file")

    if not selected_file:
        selected_file = files[0] if files else None

    attendance = pd.DataFrame()

    if selected_file:
        path = os.path.join(ATTENDANCE_FOLDER, selected_file)

        if os.path.exists(path):
            attendance = pd.read_csv(path)

    total_present = 0

    if not attendance.empty:
        total_present = len(attendance.drop_duplicates(subset=["Student"]))

    absent = total_students - total_present

    percentage = 0

    if total_students > 0:
        percentage = round((total_present / total_students) * 100, 1)

    return render_template(

        "reports.html",

        files=files,

        selected_file=selected_file,

        students=attendance.to_dict("records"),

        total_students=total_students,

        present=total_present,

        absent=absent,

        percentage=percentage,

        today=datetime.now().strftime("%d %B %Y")
    )
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/admin_dashboard")
@app.route("/admin_dashboard")
def admin_dashboard():

    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"attendance_{today}.csv"

    path = os.path.join(ATTENDANCE_FOLDER, filename)

    students = []

    if os.path.exists(path):
        df = pd.read_csv(path)
        students = df.to_dict(orient="records")

    total_students = len(face_system.students)

    present_today = len(set(student["Student"] for student in students))

    return render_template(
        "admin_dashboard.html",
        today=datetime.now().strftime("%d %B %Y"),
        total_students=total_students,
        total=present_today
    )

@app.route("/delete_student/<student_id>")
def delete_student(student_id):

    # Delete image
    for file in os.listdir("students"):
        if file.startswith(student_id + "_"):
            os.remove(os.path.join("students", file))
            break

    # Reload students
    face_system.load_students()

    return redirect(url_for("students"))

@app.route("/student_image/<filename>")
def student_image(filename):
    return send_from_directory("students", filename)

@app.route("/start_attendance")
def start_attendance_route():
    global attendance_system

    if attendance_system is None:
        attendance_system = AttendanceSystem()

        thread = threading.Thread(target=attendance_system.run)
        thread.daemon = True
        thread.start()

    return redirect(url_for("admin_dashboard"))

@app.route("/stop_attendance")
def stop_attendance_route():
    global attendance_system

    if attendance_system:
        attendance_system.stop()
        attendance_system = None

    return redirect(url_for("admin_dashboard"))

@app.route("/admin", methods=["GET", "POST"])
def admin():

    if request.method == "POST":

        institute_id = request.form["institute_id"]
        username = request.form["username"]
        password = request.form["password"]

        teachers = pd.read_csv("teachers.csv")

        teacher = teachers[
            (teachers["InstituteID"] == institute_id) &
            (teachers["Username"] == username) &
            (teachers["Password"] == password)
        ]

        if not teacher.empty:

            session["admin"] = True

            return redirect(url_for("admin_dashboard"))

        else:

            return render_template(
                "admin_login.html",
                error="Invalid Credentials"
            )

    return render_template("admin_login.html")

@app.route("/students")
def students():

    student_list = []

    for student_id, data in face_system.students.items():

        photo = ""

        for file in os.listdir("students"):
            if file.startswith(student_id + "_"):
                photo = file
                break

        student_list.append({

            "id": student_id,
            "name": data["name"],
            "photo": photo

        })

    return render_template(

        "students.html",

        students=student_list,

        total_students=len(student_list),

        today=datetime.now().strftime("%d %B %Y")

    )

@app.route("/attendance_records")
def attendance_records():

    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"attendance_{today}.csv"

    path = os.path.join(ATTENDANCE_FOLDER, filename)

    attendance = []

    if os.path.exists(path):
        df = pd.read_csv(path)
        attendance = df.to_dict(orient="records")

    total_students = len(face_system.students)

    present_today = len(set(row["Student"] for row in attendance))

    return render_template(
        "attendance_records.html",
        attendance=attendance,
        total_students=total_students,
        total=present_today,
        today=datetime.now().strftime("%d %B %Y")
    )

@app.route("/create_teacher",methods=["GET","POST"])
def create_teacher():

    if "superadmin" not in session:
        return redirect(url_for("superadmin"))

    if request.method=="POST":

        institute=request.form["institute_id"]

        teacher=request.form["teacher_name"]

        username=request.form["username"]

        password=request.form["password"]

        df=pd.read_csv("teachers.csv")

        new_teacher={

            "InstituteID":institute,

            "TeacherName":teacher,

            "Username":username,

            "Password":password

        }

        df=pd.concat([df,pd.DataFrame([new_teacher])],ignore_index=True)

        df.to_csv("teachers.csv",index=False)

        return render_template(

            "create_teacher.html",

            success="Teacher Created Successfully"

        )

    return render_template("create_teacher.html")

@app.route("/superadmin",methods=["GET","POST"])
def superadmin():

    if request.method=="POST":

        username=request.form["username"]
        password=request.form["password"]

        if username=="master" and password=="master123":

            session["superadmin"]=True

            return redirect(url_for("create_teacher"))

        return render_template(
            "superadmin_login.html",
            error="Invalid Login"
        )

    return render_template("superadmin_login.html")


# ================= DASHBOARD =================
@app.route('/dashboard')
def dashboard():

        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"attendance_{today}.csv"

        path = os.path.join(ATTENDANCE_FOLDER, filename)

        students = []

        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                students = df.to_dict(orient="records")
            except:
                students = []

        present_today = len(set(student["Student"] for student in students))   
        total_students = len(face_system.students)
        return render_template(
        'dashboard.html',
        students=students,
        total=present_today,
        total_students=total_students,
        today=datetime.now().strftime("%d %B %Y")
    )


# ================= REGISTER PAGE =================
@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':

        student_id = request.form['student_id']
        student_name = request.form['student_name']

        # Open camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Register student
        face_system.register_student(
            student_id,
            student_name,
            cap
        )

        cap.release()
        
        flash("✅ Student registered successfully!", "success")


        return redirect(url_for('dashboard'))

    return render_template('register.html')


if __name__ == '__main__':
    app.run(debug=True)