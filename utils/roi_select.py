import cv2

points = []

def on_mouse(event, x, y, flags, param):
    global points, image_show

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"point {len(points)}: ({x}, {y})")

        cv2.circle(image_show, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            image_show,
            f"({x},{y})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.imshow("image", image_show)

image = cv2.imread("picture/1.jpg")
if image is None:
    raise RuntimeError("failed to read image")

image_show = image.copy()
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", on_mouse)
cv2.imshow("image", image_show)

print("左键点击取点，按 q 退出")
while True:
    key = cv2.waitKey(20) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
print("all points:", points)