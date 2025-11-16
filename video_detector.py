import numpy as np
import cv2
import tempfile
import os
from deepfake_detector import DeepfakeDetector
from PIL import Image


def to_python(obj):
    """Recursively convert numpy types into native Python types (JSON-safe)."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_python(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj


class VideoDeepfakeDetector:
    """
    Improved video deepfake detection:
    - Full-res frame analysis
    - Face ROI detection
    - Robust temporal consistency (multi-point optical flow)
    - Confidence-weighted frame aggregation
    - JSON-safe output
    """

    def __init__(self):
        self.image_detector = DeepfakeDetector()
        self.max_frames = 48
        self.feature_count = 200
        self.lk_win_size = (21, 21)
        self.lk_max_level = 3
        self.ewma_alpha = 0.25
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # -------------------------
    # Main entry
    # -------------------------
    def analyze_video(self, video_file):
        try:
            # Save to temp if file-like
            if hasattr(video_file, "read"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(video_file.read())
                    tmp.flush()
                    video_path = tmp.name
            else:
                video_path = video_file

            frames, fps = self._extract_frames_uniform(video_path, self.max_frames)
            if not frames:
                return "REAL", 50.0, {"error": "No frames extracted"}

            frame_results = self._analyze_frames_fullres(frames)
            temporal = self._temporal_consistency(frames)
            summary = self._combine(frame_results, temporal)

            if hasattr(video_file, "read"):
                os.unlink(video_path)

            overall = summary["overall_score"]
            if overall >= 0.5:
                label = "REAL"
                conf = overall * 100.0
            else:
                label = "DEEPFAKE"
                conf = (1.0 - overall) * 100.0

            return label, conf, to_python(summary)

        except Exception as e:
            return "REAL", 55.0, {"error": f"Video analysis failed: {str(e)}"}

    # -------------------------
    # Extract frames
    # -------------------------
    def _extract_frames_uniform(self, video_path, max_frames):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], 0.0

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if total <= 0:
            frames = []
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            return frames, fps

        take = min(max_frames, total)
        indices = np.linspace(0, total - 1, take).astype(int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames, fps

    # -------------------------
    # Frame-level analysis
    # -------------------------
    def _detect_face_roi(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.1 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(rgb.shape[1], x + w + pad), min(rgb.shape[0], y + h + pad)
        return (x0, y0, x1, y1)

    def _analyze_frames_fullres(self, frames):
        indiv = []
        real_probs, avg_conf = [], []
        ewma = None

        for i, rgb in enumerate(frames):
            roi = self._detect_face_roi(rgb)
            crop = rgb if roi is None else rgb[roi[1]:roi[3], roi[0]:roi[2]]
            img = crop.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            label, conf = self.image_detector.predict(img)
            details = self.image_detector.get_analysis_details(img)

            prob_real = conf / 100.0 if label == "REAL" else (1.0 - conf / 100.0)
            ewma = prob_real if ewma is None else (
                self.ewma_alpha * prob_real + (1 - self.ewma_alpha) * ewma
            )

            indiv.append(
                {
                    "frame_index": int(i),
                    "prediction": label,
                    "confidence": float(conf),
                    "prob_real": float(prob_real),
                    "prob_real_smoothed": float(ewma),
                    "roi": None
                    if roi is None
                    else {"x0": int(roi[0]), "y0": int(roi[1]), "x1": int(roi[2]), "y1": int(roi[3])},
                    "details": to_python(details),
                }
            )
            real_probs.append(prob_real)
            avg_conf.append(conf)

        return {
            "individual_frames": indiv,
            "average_confidence": float(np.mean(avg_conf) if avg_conf else 50.0),
            "mean_prob_real": float(np.mean(real_probs) if real_probs else 0.5),
            "num_frames": int(len(indiv)),
        }

    # -------------------------
    # Temporal consistency
    # -------------------------
    def _temporal_consistency(self, frames):
        if len(frames) < 2:
            return {
                "temporal_score": 0.5,
                "motion_consistency": 0.5,
                "face_temporal_consistency": 0.5,
                "points_tracked": 0,
            }

        motion_scores, face_scores = [], []
        total_tracked = 0

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        prev_roi = self._detect_face_roi(frames[0])
        prev_mask = None
        if prev_roi is not None:
            prev_mask = np.zeros_like(prev_gray)
            x0, y0, x1, y1 = prev_roi
            prev_mask[y0:y1, x0:x1] = 255

        p0 = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.feature_count,
            qualityLevel=0.01,
            minDistance=7,
            mask=prev_mask,
        )

        if p0 is None:
            return self._farneback_temporal(frames)

        for i in range(1, len(frames)):
            next_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                next_gray,
                p0,
                None,
                winSize=self.lk_win_size,
                maxLevel=self.lk_max_level,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )

            if p1 is None or st is None:
                break

            good_new, good_old = p1[st == 1], p0[st == 1]
            total_tracked += len(good_new)

            disp = np.linalg.norm(good_new - good_old, axis=1)
            if len(disp) > 0:
                m_std, m_mean = float(np.std(disp)), float(np.mean(disp) + 1e-6)
                motion_consistency = 1.0 - min(1.0, (m_std / m_mean))
                motion_scores.append(motion_consistency)

            curr_roi = self._detect_face_roi(frames[i])
            if prev_roi is not None and curr_roi is not None:
                p_c = np.array(
                    [(prev_roi[0] + prev_roi[2]) / 2.0, (prev_roi[1] + prev_roi[3]) / 2.0]
                )
                c_c = np.array(
                    [(curr_roi[0] + curr_roi[2]) / 2.0, (curr_roi[1] + curr_roi[3]) / 2.0]
                )
                d = np.linalg.norm(c_c - p_c)
                h, w = frames[i].shape[:2]
                diag = np.hypot(h, w)
                dd = d / (diag + 1e-6)
                face_scores.append(float(1.0 - min(1.0, dd * 20.0)))

            prev_gray = next_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            prev_roi = curr_roi

        motion = float(np.mean(motion_scores) if motion_scores else 0.5)
        face_temporal = float(np.mean(face_scores) if face_scores else 0.5)
        temporal = 0.6 * motion + 0.4 * face_temporal

        return {
            "temporal_score": float(temporal),
            "motion_consistency": float(motion),
            "face_temporal_consistency": float(face_temporal),
            "points_tracked": int(total_tracked),
        }

    def _farneback_temporal(self, frames):
        flows = []
        for i in range(1, len(frames)):
            prev = cv2.cvtColor(frames[i - 1], cv2.COLOR_RGB2GRAY)
            nxt = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev,
                nxt,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=21,
                iterations=3,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flows.append(float(np.std(mag) / (np.mean(mag) + 1e-6)))

        if not flows:
            return {
                "temporal_score": 0.5,
                "motion_consistency": 0.5,
                "face_temporal_consistency": 0.5,
                "points_tracked": 0,
            }

        inv = [1.0 - min(1.0, v) for v in flows]
        score = float(np.mean(inv))
        return {
            "temporal_score": score,
            "motion_consistency": score,
            "face_temporal_consistency": 0.5,
            "points_tracked": 0,
        }

    # -------------------------
    # Combine temporal + frame data
    # -------------------------
    def _combine(self, frame_results, temporal):
        probs, weights = [], []
        for fr in frame_results["individual_frames"]:
            probs.append(fr["prob_real_smoothed"])
            weights.append(max(1e-3, fr["confidence"] / 100.0))

        weighted_prob_real = (
            float(np.average(probs, weights=weights)) if probs else 0.5
        )
        temporal_score = temporal.get("temporal_score", 0.5)
        overall = 0.75 * weighted_prob_real + 0.25 * temporal_score

        return {
            "overall_score": float(overall),
            "frame_analysis": frame_results,
            "temporal_analysis": temporal,
            "summary": {
                "Frame Consistency": frame_results["mean_prob_real"],
                "Temporal Consistency": temporal_score,
                "Motion Analysis": temporal.get("motion_consistency", 0.5),
                "Face Tracking": temporal.get("face_temporal_consistency", 0.5),
            },
        }
