import cv2
import face_recognition
import pickle
import datetime
import csv

with open("models/encodings.pickle", "rb") as f:
    data = pickle.load(f)

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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding, (top, right, bottom, left) in zip(encodings, boxes):
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        if True in matches:
            best_match = matches.index(True)
            name = data["names"][best_match]
            if name not in marked_today:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(attendance_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, timestamp])
                marked_today.add(name)

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()