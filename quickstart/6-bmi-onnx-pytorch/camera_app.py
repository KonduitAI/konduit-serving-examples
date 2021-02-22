import numpy as np
import cv2
import requests

from urllib3.exceptions import InsecureRequestWarning

width = 1280
height = 720
out_width = 320
out_height = 240
fps = 1
camera_index = 0

url = 'https://202.165.22.122/infer'
# url = 'http://localhost:8082/infer'

cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # set Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # set Height
cap.set(cv2.CAP_PROP_FPS, fps)
font = cv2.FONT_HERSHEY_DUPLEX

# Suppress only the single warning from urllib3 needed.
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

images = [0 for i in range(fps)]


def render_image(index, output):
    print("Response at {}: {}".format(index, output))
    boxes = np.asarray([output['boxes']])
    if len(boxes[0]) == 0:
        return

    box = boxes[0]
    out_size = 112
    img = images[index % fps].copy()
    _height, _width, _ = img.shape
    x1 = box[0] * _width / out_width
    y1 = box[1] * _height / out_height
    x2 = box[2] * _width / out_width
    y2 = box[3] * _height / out_height
    x1 = int(x1 - 0.1 * x1)
    y1 = int(y1 - 0.1 * y1)
    x2 = int(x2 + 0.1 * x2)
    y2 = int(y2 + 0.1 * y2)
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    size = int(max([w, h]) * 1.1)
    cx = x1 + w // 2
    cy = y1 + h // 2
    x1 = cx - size // 2
    x2 = x1 + size
    y1 = cy - size // 2
    y2 = y1 + size
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(_width, x2)
    y2 = min(_height, y2)

    cv2.rectangle(images[index % fps], (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(images[index % fps], str(output['bmi_class']), (x1 + 5, y1 - 5), font, 1, (255, 255, 255), 2)
    cv2.putText(images[index % fps], str(output['bmi_value']), (x1 + 5, y2), font, 1, (255, 255, 0), 1)

    cv2.imshow('video', images[index % fps])


def send_request(index):
    im_encoded = cv2.imencode(".jpg", cv2.resize(images[index % fps], (out_width, out_height)))[1]
    encoded_string = im_encoded.tostring()
    files = {'image': ('image.jpg', encoded_string, 'image/jpeg', {'Expires': '0'})}
    response = requests.post(url, files=files, verify=False).json()
    render_image(index, response)


def run():
    i = -1
    while True:
        i += 1
        ret, orig_image = cap.read()
        images[i % fps] = orig_image.copy()

        send_request(i)

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


run()


