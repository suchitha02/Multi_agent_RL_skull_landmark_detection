import numpy as np

MM_PER_PX = 0.1
SDR_THRESHOLDS = (2.0, 2.5, 3.0, 4.0)


def radial_errors(pred, gt):
    """(N,K) radial errors in mm. NaN where ground truth is missing."""
    return np.linalg.norm(np.asarray(pred) - np.asarray(gt), axis=-1) * MM_PER_PX


def summarize(err, landmark_names=None):
    """MRE, SD, SDR at the standard thresholds, and per-landmark MRE (all in mm / %)."""
    flat = err[~np.isnan(err)]
    out = {"MRE_mm": float(flat.mean()), "SD_mm": float(flat.std()), "n_points": int(flat.size)}
    for t in SDR_THRESHOLDS:
        out[f"SDR_{t}mm_%"] = float((flat <= t).mean() * 100)
    per = np.nanmean(err, 0)
    names = landmark_names or [str(i) for i in range(err.shape[1])]
    out["per_landmark_MRE_mm"] = {n: (None if np.isnan(v) else float(v)) for n, v in zip(names, per)}
    return out


def per_image_mre(err):
    return np.nanmean(err, 1)
