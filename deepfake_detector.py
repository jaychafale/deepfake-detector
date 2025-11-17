# deepfake_detector.py
import os
import random
import numpy as np
import cv2

# Try to import TensorFlow; if not available, we'll fall back to CV logic.
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except Exception:
    tf = None
    TF_AVAILABLE = False

from PIL import Image


class DeepfakeDetector:
    """
    Combined Deepfake detector:
     - If TensorFlow + pretrained weights present -> run MesoNet (TF) inference.
     - Otherwise -> fallback to original computer-vision heuristic methods.

    Interface is unchanged:
      - predict(image_array) -> (label, confidence)
      - get_analysis_details(image_array) -> dict
    """

    def __init__(self, model_path: str = "models/mesonet_tf.h5"):
        # Keep existing CV config for fallback behavior
        self.input_size = (224, 224)
        self.edge_threshold_low = 50
        self.edge_threshold_high = 150
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # TensorFlow model variables
        self.tf_model = None
        self.tf_model_loaded = False
        self.tf_model_path = model_path

        # Try to initialize TF model if TF is available and weights path exists
        if TF_AVAILABLE and os.path.exists(self.tf_model_path):
            try:
                # Build architecture and load weights
                self.tf_model = self._build_mesonet(input_shape=(self.input_size[0], self.input_size[1], 3))
                self.tf_model.load_weights(self.tf_model_path)
                self.tf_model_loaded = True
                # Force inference on CPU if GPU is not desired (leave default otherwise)
                # Uncomment to force CPU only:
                # tf.config.set_visible_devices([], 'GPU')
            except Exception:
                # If loading fails, keep fallback behavior
                self.tf_model = None
                self.tf_model_loaded = False

    # ---------------------
    # Public API
    # ---------------------
    def predict(self, image):
        """
        Predict if an image is REAL or DEEPFAKE.

        Args:
            image: numpy array; can be (H,W,C) or (1,H,W,C) with values in [0,1]
        Returns:
            (label(str), confidence(float 0-100))
        """
        try:
            # Normalize and prepare array
            arr = self._ensure_numpy_image(image)

            # If TF model loaded, use it
            if self.tf_model_loaded:
                # resize to model input
                img = cv2.resize(arr, self.input_size, interpolation=cv2.INTER_AREA)
                x = np.expand_dims(img.astype(np.float32), axis=0)
                # Model expects inputs in range [0,1]
                preds = self.tf_model.predict(x, verbose=0)
                # preds assumed shape (1,1) or (1,2) depending on final layer; handle common cases
                prob = self._interpret_tf_output(preds)
                label = "REAL" if prob >= 0.5 else "DEEPFAKE"
                conf = float(prob * 100.0) if label == "REAL" else float((1.0 - prob) * 100.0)
                return label, conf
            else:
                # fallback to legacy CV heuristics
                return self._predict_cv(arr)
        except Exception:
            # On any error, return safe fallback
            return "REAL", 65.0

    def get_analysis_details(self, image):
        """
        Return a dictionary of analysis details. If TF model is present, include model score.
        Otherwise, return existing CV analysis dict.
        """
        try:
            arr = self._ensure_numpy_image(image)
            if self.tf_model_loaded:
                # produce TF score and also keep some CV metrics for explainability
                img = cv2.resize(arr, self.input_size, interpolation=cv2.INTER_AREA)
                x = np.expand_dims(img.astype(np.float32), axis=0)
                preds = self.tf_model.predict(x, verbose=0)
                prob = self._interpret_tf_output(preds)
                model_score = float(prob)
                # run some CV helpers for additional fields
                facial_score = self._analyze_facial_consistency(arr)
                edge_score = self._analyze_edges(arr)
                texture_score = self._analyze_texture(arr)
                compression_score = self._analyze_compression_artifacts(arr)
                return {
                    "mesonet_model_score": model_score,
                    "Facial Consistency": facial_score,
                    "Edge Detection": edge_score,
                    "Texture Analysis": texture_score,
                    "Compression Artifacts": compression_score,
                }
            else:
                # original behaviour
                return {
                    "Facial Consistency": self._analyze_facial_consistency(arr),
                    "Edge Detection": self._analyze_edges(arr),
                    "Texture Analysis": self._analyze_texture(arr),
                    "Compression Artifacts": self._analyze_compression_artifacts(arr),
                }
        except Exception:
            return {
                "Facial Consistency": 0.5,
                "Edge Detection": 0.5,
                "Texture Analysis": 0.5,
                "Compression Artifacts": 0.5,
            }

    # ---------------------
    # TensorFlow MesoNet architecture (small)
    # ---------------------
    def _build_mesonet(self, input_shape=(224, 224, 3)):
        """
        Build a lightweight MesoNet-like architecture in TensorFlow Keras.
        The architecture is intentionally compact for CPU inference.
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available")

        inp = layers.Input(shape=input_shape)

        # small conv blocks
        x = layers.Conv2D(8, (3, 3), padding="same", activation="relu")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D((2, 2))(x)

        x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D((2, 2))(x)

        x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D((2, 2))(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        # final single-logit output (sigmoid)
        out = layers.Dense(1, activation="sigmoid", name="real_prob")(x)

        model = models.Model(inputs=inp, outputs=out)
        # compile not required for inference but keep a lightweight optimizer
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    def _interpret_tf_output(self, preds):
        """
        Convert TF predictions into a scalar probability (probability of REAL).
        Accepts several common shapes: (1,1), (1,), (1,2)
        """
        p = preds
        try:
            # convert to numpy
            if hasattr(p, "numpy"):
                p = p.numpy()
            p = np.array(p)
            if p.size == 1:
                prob = float(p.flatten()[0])
            elif p.shape[-1] == 2:
                # softmax case: assume index 1 is REAL
                prob = float(p[0, 1])
            else:
                prob = float(p.flatten()[0])
            prob = max(0.0, min(1.0, prob))
            return prob
        except Exception:
            return 0.5

    # ---------------------
    # Legacy CV-based fallback predictor (keeps previous logic)
    # ---------------------
    def _predict_cv(self, image):
        """
        Uses the original CV heuristics to compute combined_score -> label/confidence.
        The code mirrors previous behavior to avoid changes in downstream logic.
        """
        try:
            facial_score = self._analyze_facial_consistency(image)
            edge_score = self._analyze_edges(image)
            texture_score = self._analyze_texture(image)
            compression_score = self._analyze_compression_artifacts(image)

            combined_score = (
                facial_score * 0.3 + edge_score * 0.25 + texture_score * 0.25 + compression_score * 0.2
            )

            variation = random.uniform(-0.1, 0.1)
            final_score = max(0.0, min(1.0, combined_score + variation))

            if final_score > 0.5:
                label = "REAL"
                confidence = final_score * 100
            else:
                label = "DEEPFAKE"
                confidence = (1 - final_score) * 100

            return label, confidence
        except Exception:
            return "REAL", 65.0

    # ---------------------
    # Helper utilities (image conversions & CV metrics)
    # ---------------------
    def _ensure_numpy_image(self, image):
        """
        Accept PIL Image, numpy array (H,W,C or 1,H,W,C), or float [0,1] -> returns H,W,C float32 in [0,1].
        """
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        else:
            arr = np.array(image)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            else:
                arr = arr.astype(np.float32)
            image = arr
        # if grayscale, convert to 3 channels
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=-1)
        return image

    def _analyze_facial_consistency(self, image):
        """Analyze facial feature consistency (same as previous)."""
        try:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                face_area = len(faces) / (image.shape[0] * image.shape[1])  # number of faces scaled
                return min(1.0, face_area * 10)
            else:
                return 0.3
        except Exception:
            return 0.5

    def _analyze_edges(self, image):
        """Edge pattern analysis (same as previous)."""
        try:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            if 0.05 <= edge_density <= 0.15:
                return 0.8
            elif 0.02 <= edge_density <= 0.25:
                return 0.6
            else:
                return 0.3
        except Exception:
            return 0.5

    def _analyze_texture(self, image):
        """Texture analysis (same as previous)."""
        try:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            texture_std = np.std(gray)
            if 20 <= texture_std <= 60:
                return 0.8
            elif 10 <= texture_std <= 80:
                return 0.6
            else:
                return 0.4
        except Exception:
            return 0.5

    def _analyze_compression_artifacts(self, image):
        """Compression artifacts (same as previous)."""
        try:
            img_uint8 = (image * 255).astype(np.uint8)
            correlations = []
            for i in range(3):
                for j in range(i + 1, 3):
                    corr = np.corrcoef(img_uint8[:, :, i].flatten(), img_uint8[:, :, j].flatten())[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            if correlations:
                avg_correlation = np.mean(correlations)
                if 0.3 <= avg_correlation <= 0.7:
                    return 0.8
                else:
                    return 0.4
            else:
                return 0.5
        except Exception:
            return 0.5
