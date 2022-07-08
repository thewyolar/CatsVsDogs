import httplib2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import load_model


def resize_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image, label


def download_image(link):
    h = httplib2.Http('.cache')
    response, content = h.request(link)
    out = open('test.jpg', 'wb')
    out.write(content)
    out.close()


if __name__ == '__main__':
    model = load_model("cats_vs_dogs_mnist")
    download_image("https://snitsya-son.ru/uploads/bonica/2019/09/sobaka--luchshiy-drug-cheloveka.jpg")
    img = tf.keras.preprocessing.image.load_img('test.jpg')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_resized, _ = resize_image(img_array, __name__)
    img_expended = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_expended)[0][0]
    pred_label = 'КОТ' if prediction < 0.5 else 'СОБАКА'
    plt.figure()
    plt.imshow(img)
    plt.title(f'{pred_label} {prediction}')
    plt.show()
