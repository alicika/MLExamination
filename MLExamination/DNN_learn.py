#coding:UTF-8

import numpy as np
from keras import models
from keras.utils import to_categorical
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.datasets import mnist


#準備


(x_train, y_train), (x_test, y_test) = mnist.load_data()

#データ形式をDNNモデルに合わせる（3次元のndarrayから2次元のndarrayへ変更）
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

#データを0.0～1.0へ正規化
x_train = x_train.astype('float32')/ 255.0
x_test = x_test.astype('float32')/ 255.0

#ラベルをone-hot表現に変換
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#訓練データを訓練データと評価データに分割
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=1)


# 学習


#モデルの定義
model = models.Sequential()

#入力部
model.add(Dense(16, activation='relu', input_shape=(28 * 28, )))

#出力部
model.add(Dense(10, activation='softmax'))

#学習プロセスの設定
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#モデルの訓練
history = model.fit(x_train, y_train, batch_size=512, epochs=10, validation_data=(x_valid, y_valid))


# モデルの実験


#先程の学習結果に影響を及ぼさないように model2 という変数名にする
from keras import models
from keras.layers import Dense

model2 = models.Sequential()

model2.add(Dense(16, activation='relu', input_shape=(784,)))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(10, activation='softmax'))

model2.summary()


#モデルと重みを保存
open('model.json', 'w').write(model.to_json())
model.save_weights('weights.h5')

# テストデータで評価
eval_test = model.evaluate(x_test, y_test)
print("Evaluate : " + str(eval_test))

# F1スコアで評価
y_pred = model.predict(x_test)
eval_f1 = f1_score(np.argmax(y_test, 1), np.argmax(y_pred, 1), average='macro')
print("F1 score : " + str(eval_f1))


#Loss
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

#Accuracy
train_acc = history.history['acc']
valid_acc = history.history['val_acc']
nb_epoch = len(train_acc)
plt.plot(range(nb_epoch), train_acc, marker='.', label='train_acc')
plt.plot(range(nb_epoch), valid_acc, marker='.', label='valid_acc')
plt.legend(loc='best', fontsize=10)


# グラフ出力


plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()