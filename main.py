import cv2
import time

first_frame = None
video_capture = cv2.VideoCapture(0)

while True:
    check, frame = video_capture.read()

    gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_video = cv2.GaussianBlur(gray_video, (15, 15), 3)

    if first_frame is None:
        first_frame = gray_video
        continue

    delta_frame = cv2.absdiff(first_frame, gray_video)
    thresh_frame = cv2.threshold(delta_frame, 100, 210, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 700:  # устанавливаем минимальный размер контуров
            continue
        (x, y, w, h) = cv2.boundingRect(contour)  # задаем размеры обводки
        cv2.rectangle(frame, ((x - 20), y), (x + 200, x - h), (0, 255, 255), 3) 

    cv2.imshow('Blur frame', gray_video)
    cv2.imshow("Delta frame", delta_frame)
    cv2.imshow("Threshold delta", thresh_frame)
    cv2.imshow("Color frame", frame)

    key = cv2.waitKey(1)
    print(gray_video)
    print(delta_frame)

    if key == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
