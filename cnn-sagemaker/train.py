"""Train CNN (TensorFlow or PyTorch). Works locally and as a SageMaker training script."""
import argparse, os
import numpy as np
from models import build_keras_cnn, TorchCNN
from evaluate import evaluate

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def train_tf(epochs, batch, model_dir):
    import tensorflow as tf
    (xtr, ytr), (xte, yte) = tf.keras.datasets.cifar10.load_data()
    xtr, xte = xtr / 255.0, xte / 255.0
    model = build_keras_cnn()
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(xtr, ytr, epochs=epochs, batch_size=batch, validation_split=0.1)
    evaluate(yte.ravel(), model.predict(xte).argmax(1), CLASSES)
    # SageMaker TF Serving expects <model_dir>/<version>/
    model.export(os.path.join(model_dir, "1"))


def train_torch(epochs, batch, model_dir):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    tf_ = transforms.ToTensor()
    tr = datasets.CIFAR10("data", True, tf_, download=True)
    te = datasets.CIFAR10("data", False, tf_, download=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = TorchCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for x, y in DataLoader(tr, batch, shuffle=True):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
        print(f"epoch {ep + 1}/{epochs} loss={loss.item():.4f}")
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for x, y in DataLoader(te, 256):
            preds += model(x.to(dev)).argmax(1).cpu().tolist(); trues += y.tolist()
    evaluate(np.array(trues), np.array(preds), CLASSES)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=["tf", "torch"], default="tf")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    a = ap.parse_args()
    (train_tf if a.framework == "tf" else train_torch)(a.epochs, a.batch, a.model_dir)
