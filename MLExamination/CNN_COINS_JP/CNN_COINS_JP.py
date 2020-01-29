#coding:UTF-8
import numpy as np
import keras
from keras import models
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import f1_score
import glob

# ==================================================
# 学習データ準備
# ==================================================
# 対象のフォルダの中にある画像を順次読み込む
list_folder = ["0_none", "1_1yen", "2_5yen", "3_10yen", "4_50yen", "5_100yen", "6_500yen"]
x_data = []
y_data = []
for num, folder in enumerate(list_folder):
    print("Load images... " + str(num))
    list_img = glob.glob(folder + "/*.jpg")
    for img in list_img:
        img = load_img(img, target_size=(32, 32))
        img = img_to_array(img)
        x_data.append(img)
        y_data.append(num)

# データ形式をCNNモデルに合わせる
x_data = np.array(x_data)
y_data = np.array(y_data)

# データを0.0～1.0へ正規化
x_data = x_data.astype('float32') / 255.0

# ラベルをone-hot表現に変換
y_data = to_categorical(y_data, len(list_folder))

# データを訓練データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

# 訓練データを訓練データと評価データに分割
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

# ===================================================
# 学習
# ===================================================
# モデルの定義
model = models.Sequential()

# 入力部
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 中間部1
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 中間部2
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

# 中間部3
model.add(Flatten())
model.add(Dense(64, activation='relu'))

# 出力部
model.add(Dense(7, activation='softmax'))  # ★

# 学習プロセスの設定
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# モデルの訓練
history = model.fit(x_train, y_train, batch_size=16, epochs=20,
                    validation_data=(x_valid, y_valid))  # ★batch_size 128 > 16

# ===================================================
# モデルと重みの保存
# ===================================================
open('model_coin.json', 'w').write(model.to_json())  # ★
model.save_weights('weights_coin.h5')  # ★

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