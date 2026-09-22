"""
DHWANI - DS-CNN Wakeword Detection Model
Wakeword: "HEY ISRO" vs "NOT HEY ISRO"
"""

import os
import glob
import argparse
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split

SR = 16000
DURATION = 1  # 1 second audio
N_MELS = 40
N_FFT = 512
HOP = 400


def extract_features(file_path):
    """Load audio file and convert to 40x40 log-mel spectrogram."""
    audio, _ = librosa.load(file_path, sr=SR, mono=True, duration=DURATION)
    audio = librosa.util.fix_length(audio, size=SR)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    # Min-max normalization
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

    # Ensure shape (40, 40)
    mel = mel[:, :40]
    if mel.shape[1] < 40:
        mel = np.pad(mel, ((0, 0), (0, 40 - mel.shape[1])))

    return mel


def load_dataset(dataset_path, classes=None):
    """Load and process audio files from class directories."""
    if classes is None:
        classes = ["01_HEY_ISRO", "02_NOT_HEY_ISRO"]

    X = []
    y = []

    for label, folder in enumerate(classes):
        folder_path = os.path.join(dataset_path, folder)
        files = glob.glob(os.path.join(folder_path, "**", "*.wav"), recursive=True)
        print(f"[{folder}] Found {len(files)} WAV files")

        for f in files:
            try:
                feat = extract_features(f)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print(f"Skipped {f}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    X = X[..., np.newaxis]  # (N, 40, 40, 1)

    print(f"\nDataset loaded: {X.shape}, Labels: {y.shape}")
    return X, y


def DSBlock(x, filters, stride):
    """Depthwise Separable Convolution Block."""
    x = tf.keras.layers.DepthwiseConv2D(
        3, strides=stride, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        filters, 1, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def build_ds_cnn_model(input_shape=(40, 40, 1), num_classes=2, dropout_rate=0.35):
    """Constructs the DS-CNN Wakeword architecture."""
    inputs = tf.keras.Input(shape=input_shape)

    # Feature augmentation
    x = tf.keras.layers.GaussianNoise(0.03)(inputs)

    # Initial standard convolution
    x = tf.keras.layers.Conv2D(
        16, 3, strides=2, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Depthwise Separable CNN blocks
    x = DSBlock(x, 16, 1)
    x = DSBlock(x, 24, 2)
    x = DSBlock(x, 32, 2)
    x = DSBlock(x, 48, 2)

    # Pooling and classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="DHWANI_DS_CNN")
    return model


class StopAtTargetAccuracy(tf.keras.callbacks.Callback):
    """Stop training when accuracy threshold is met."""
    def __init__(self, target_accuracy=0.98):
        super().__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("accuracy", 0)
        if acc >= self.target_accuracy:
            print(f"\nTarget training accuracy reached: {acc * 100:.2f}%. Stopping training.")
            self.model.stop_training = True


def main():
    parser = argparse.ArgumentParser(description="Train DHWANI DS-CNN Wakeword Model")
    parser.add_argument("--dataset_path", type=str, default="ISRO_WakeWord_WAV", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--output_model", type=str, default="dhwani_ds_cnn.keras", help="Path to save trained model")
    parser.add_argument("--target_acc", type=float, default=0.98, help="Early stop accuracy threshold")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' not found.")
        print("Please provide a valid dataset path containing '01_HEY_ISRO' and '02_NOT_HEY_ISRO'.")
        return

    X, y = load_dataset(args.dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = build_ds_cnn_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True),
        StopAtTargetAccuracy(target_accuracy=args.target_acc)
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("\n===================================")
    print("       DHWANI TRAINING RESULT      ")
    print("===================================")
    print(f"Test Accuracy : {accuracy * 100:.2f}%")
    print(f"Test Loss     : {loss:.4f}")
    print("===================================")

    if args.output_model:
        model.save(args.output_model)
        print(f"Model saved to {args.output_model}")


if __name__ == "__main__":
    main()
