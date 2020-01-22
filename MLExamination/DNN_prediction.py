#coding:UTF-8

import numpy as np
import keras
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.datasets import mnist
import random


#===================================================
#モデル、重み、学習プロセスの読み込み
#===================================================
#モデルを読み込む
model = model_from_json(open('model.json').read())

#重みを読み込む
model.load_weights('weights.h5')

#損失関数、オプティマイザを指定
model.compile(loss='categorical_crossentropy', optimizer='adam')


#===================================================
#推論する画像データ読み込み
#===================================================
#全画像データ読み込み
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#推論する画像をランダムに指定
img_num = random.randint(0, 9999) #★
img = x_test[img_num]

#確認のため、画像表示
plt.figure(figsize=(2, 2))
plt.axis("off")
plt.title(str(img_num))
plt.imshow(img)

#データ形式をDNNモデルに合わせる（2次元のndarrayから1次元のndarrayへ変更）
x = img.reshape(28 * 28)

#データを0.0～1.0へ正規化
x = x.astype('float32')/ 255.0

#次元を合わせる
x = np.expand_dims(x, axis=0)


#===================================================
#推論と結表示
#===================================================
#推論
preds = model.predict(x)
print("predicts : " + str(preds))

#predsのインデックスでソート
preds_argsort = np.argsort(preds)
print("sort : " + str(preds_argsort))

#最大のインデックス
index = preds_argsort[0][-1]
print("index(max) : " + str(index))

#推論結果をリストから表示
label_list = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
print("予測 : " +str(label_list[index]))

#推論結果の確率を表示
probability = preds[0][index] * 100
print("確率 : " + str(probability) + " %")