import cv2
import pickle
import csv
import datetime

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/trainer.yml")

with open("models/labels.pickle", "rb") as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}

attendance_file = "attendance.csv"
try:
    with open(attendance_file, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Timestamp"])
except FileExistsError:
    pass

cap = cv2.VideoCapture(0)
marked_today = set()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            name = labels[id_]
            if name not in marked_today:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(attendance_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, timestamp])
                marked_today.add(name)

            cv2.putText(frame, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("LBPH Attendance", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()