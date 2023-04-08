import cv2
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt


model_url = "https://tfhub.dev/intel/midas/v2_1_small/1"
midas = hub.load(model_url)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('CV2Frame', frame)
    img_tensor = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

    with tf.device('/CPU:0'):
        prediction = midas(img_tensor)
        prediction = tf.image.resize(prediction, img.shape[:2], method='bicubic', antialias=True)
        output = prediction.numpy()[0, :, :, 0]
        plt.imshow(output)
        plt.pause(0.000001)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

plt.show()













