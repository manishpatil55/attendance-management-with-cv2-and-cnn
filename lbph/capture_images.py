import cv2
import os

person_name = input("Enter the name of the person: ")
dataset_path = os.path.join("dataset", person_name)
os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Capture - Press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    file_path = os.path.join(dataset_path, f"{count}.jpg")
    cv2.imwrite(file_path, gray)
    count += 1

cap.release()
cv2.destroyAllWindows()
print(f"Images saved to {dataset_path}")