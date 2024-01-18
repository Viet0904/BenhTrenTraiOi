import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Đường dẫn đến thư mục chứa ảnh
data_path = "./x84p2g3k6z-1/Guava_Dataset_Original_Image"
augmented_data_path = "./x84p2g3k6z-1/Guava_Disease_Augmented_Image"  # Thay đổi đường dẫn tới thư mục tập dữ liệu tăng cường

# Kích thước bạn muốn sử dụng cho hình ảnh
image_width = 214
image_height = 214 


# Hàm đọc dữ liệu với mã hóa one-hot
def load_data_one_hot(data_path):
    X = []  # Danh sách để lưu trữ ảnh
    y = []  # Danh sách để lưu trữ nhãn

    label_encoder = LabelEncoder()

    # Duyệt qua từng thư mục trong thư mục chứa dữ liệu
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)

        # Duyệt qua từng tệp trong thư mục con
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)

            # Đọc ảnh và chuyển về kích thước cần thiết
            img = cv2.imread(img_path)
            img = cv2.resize(img, (image_width, image_height))

            # Thêm ảnh và nhãn vào danh sách
            X.append(img)
            y.append(label)

    # Mã hóa nhãn one-hot
    y_encoded = label_encoder.fit_transform(y)
    y_one_hot = to_categorical(y_encoded)

    return np.array(X), y_one_hot


# Load dữ liệu gốc
X, y_one_hot = load_data_one_hot(data_path)

# Load dữ liệu tăng cường
X_augmented, y_augmented = load_data_one_hot(augmented_data_path)

# Kết hợp dữ liệu gốc và tăng cường
X_combined = np.concatenate((X, X_augmented), axis=0)
y_combined = np.concatenate((y_one_hot, y_augmented), axis=0)

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42
)

# Lấy kích thước của ảnh từ tập train
input_shape = X_train[0].shape


# Tạo mô hình
def create_advanced_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model


# Số lớp (bệnh trên trái ổi) trong tập dữ liệu của bạn
num_classes = 5

# Tạo mô hình
advanced_model = create_advanced_model(input_shape, num_classes)

# Biên soạn mô hình với hàm loss là "categorical_crossentropy"
advanced_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Tạo generator để thực hiện augmentation dữ liệu cho tập train
datagen_train = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)

# Tạo generator để thực hiện augmentation dữ liệu cho tập test (không cần horizontal_flip)
datagen_test = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

# Huấn luyện mô hình với số epochs lớn hơn và sử dụng fit_generator
history = advanced_model.fit_generator(
    datagen_train.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,  # Số batch mỗi epoch
    epochs=10,  # Số lượng epochs
    validation_data=datagen_test.flow(X_test, y_test),
)

# Đánh giá mô hình trên tập test
test_loss, test_acc = advanced_model.evaluate(X_test, y_test)
print("Advanced Model Test accuracy:", test_acc)
