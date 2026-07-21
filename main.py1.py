import cv2
import numpy as np

# قراءة الصورة
image = cv2.imread("test.jpg")

if image is None:
    print("Image not found!")
    exit()

# تحويل الصورة إلى HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# نطاقات الألوان
colors = {
    "Red": ([0, 120, 70], [10, 255, 255]),
    "Green": ([36, 50, 70], [89, 255, 255]),
    "Blue": ([90, 50, 70], [130, 255, 255]),
    "Yellow": ([20, 100, 100], [35, 255, 255])
}

# البحث عن كل لون
for color, (lower, upper) in colors.items():

    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:

        if cv2.contourArea(cnt) > 300:

            x, y, w, h = cv2.boundingRect(cnt)

            # رسم مربع حول الجسم
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # كتابة اسم اللون
            cv2.putText(
                image,
                color,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

# عرض الصورة
cv2.imshow("Color Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()