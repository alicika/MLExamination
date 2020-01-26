#coding:UTF-8
import numpy as np
import keras
from keras import models
from keras.utils import to_categorical
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import f1_score
from keras.datasets import cifar10

# ===================================================
# 学習データ準備
# ===================================================
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

# データ形式をDNNモデルに合わせる（4次元のndarrayから2次元のndarrayへ変更）
x_train = x_train.reshape(50000, 32 * 32 * 3) 
x_test = x_test.reshape(10000, 32 * 32 * 3) 

# データを0.0～1.0へ正規化
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# ラベルをone-hot表現に変換
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 訓練データを訓練データと評価データに分割
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

# ===================================================
# 学習
# ===================================================
# モデルの定義
model = models.Sequential()

# 入力部
model.add(Dense(512, activation='relu', input_shape=(32 * 32 * 3,)))  # ★

# 出力部
model.add(Dense(10, activation='softmax'))

# 学習プロセスの設定
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# モデルの訓練
history = model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_valid, y_valid))

# ===================================================
# モデルと重みの保存
# ===================================================
open('model.json', 'w').write(model.to_json())
model.save_weights('weights.h5')

# ===================================================
# グラフ出力
# ===================================================
# Loss
train_loss = history.history['loss']
valid_loss = history.history['val_loss']
nb_epoch = len(train_loss)
plt.plot(range(nb_epoch), train_loss, marker='.', label='train_loss')
plt.plot(range(nb_epoch), valid_loss, marker='.', label='valid_loss')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Accuracy
train_acc = history.history['acc']
valid_acc = history.history['val_acc']
nb_epoch = len(train_acc)
plt.plot(range(nb_epoch), train_acc, marker='.', label='train_acc')
plt.plot(range(nb_epoch), valid_acc, marker='.', label='valid_acc')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# ==================================================
# 評価
# ==================================================
# テストデータ
eval_test = model.evaluate(x_test, y_test)
print("Evaluate : " + str(eval_test))

# F値
y_pred = model.predict(x_test)
eval_f1 = f1_score(np.argmax(y_test, 1), np.argmax(y_pred, 1), average='macro')
print("F1 score : " + str(eval_f1))