"""Pre-download datasets used in chapters 1-17 of Hands-On ML so they're
cached locally for offline use (e.g. on a plane).

Run with: uv run python scripts/prefetch_datasets.py
"""
import os
import tarfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def log(msg):
    print(f"[prefetch] {msg}")


def fetch_housing():
    target = DATA_DIR / "housing"
    if (target / "housing.csv").exists():
        log("housing.csv already cached")
        return
    log("downloading housing dataset...")
    url = "https://github.com/ageron/data/raw/main/housing.tgz"
    tgz_path = DATA_DIR / "housing.tgz"
    urllib.request.urlretrieve(url, tgz_path)
    with tarfile.open(tgz_path) as f:
        f.extractall(DATA_DIR)
    tgz_path.unlink()
    log("housing dataset ready")


def fetch_mnist():
    log("downloading MNIST via sklearn.fetch_openml (cached in scikit_learn_data)...")
    from sklearn.datasets import fetch_openml
    fetch_openml("mnist_784", version=1, as_frame=False, data_home=DATA_DIR / "sklearn_data")
    log("MNIST ready")


def fetch_olivetti_faces():
    log("downloading Olivetti faces via sklearn...")
    from sklearn.datasets import fetch_olivetti_faces
    fetch_olivetti_faces(data_home=DATA_DIR / "sklearn_data")
    log("Olivetti faces ready")


def fetch_keras_datasets():
    log("downloading Keras built-in datasets (fashion_mnist, cifar10, imdb)...")
    import tensorflow as tf
    tf.keras.datasets.fashion_mnist.load_data()
    log("fashion_mnist ready")
    try:
        tf.keras.datasets.cifar10.load_data()
        log("cifar10 ready")
    except Exception as e:
        log(f"cifar10 download failed ({e}), retrying with browser user-agent...")
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        tf.keras.datasets.cifar10.load_data()
        log("cifar10 ready")
    tf.keras.datasets.imdb.load_data()
    log("imdb ready")


def fetch_shakespeare():
    target = DATA_DIR / "shakespeare.txt"
    if target.exists():
        log("shakespeare.txt already cached")
        return
    log("downloading Shakespeare text corpus...")
    url = "https://homl.info/shakespeare"
    urllib.request.urlretrieve(url, target)
    log("shakespeare corpus ready")


if __name__ == "__main__":
    fetch_housing()
    fetch_mnist()
    fetch_olivetti_faces()
    fetch_keras_datasets()
    fetch_shakespeare()
    log("all datasets cached, ready for offline use")
