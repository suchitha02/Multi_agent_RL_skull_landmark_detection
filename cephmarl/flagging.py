"""Uncertainty flagging: shape residual model, combining signals, and flag metrics."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


# ---------------------------------------------------------------- shape model
def _align(x, ref):
    """Similarity-align shape x (K,2) to ref (K,2) (no reflection)."""
    xc, rc = x - x.mean(0), ref - ref.mean(0)
    u, s, vt = np.linalg.svd(xc.T @ rc)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1, d]) @ vt
    scale = (s * [1, d]).sum() / (xc ** 2).sum()
    return scale * xc @ rot + ref.mean(0)


class ShapeModel:
    """Gaussian over Procrustes-aligned shapes; residual of one landmark given all others."""

    def __init__(self, mean, cov):
        self.mean, self.cov = np.asarray(mean), np.asarray(cov)
        K = self.mean.shape[0]
        mu, S = self.mean.reshape(-1), self.cov
        self.cond = []
        for i in range(K):
            a = [2 * i, 2 * i + 1]
            o = [j for j in range(2 * K) if j not in a]
            gain = S[np.ix_(a, o)] @ np.linalg.inv(S[np.ix_(o, o)])
            ci = S[np.ix_(a, a)] - gain @ S[np.ix_(o, a)]
            self.cond.append((a, o, gain, np.linalg.inv(ci)))
        self.mu = mu

    @classmethod
    def fit(cls, shapes, iters=3, shrink=0.1):
        shapes = np.asarray(shapes, dtype=np.float64)
        shapes = shapes[~np.isnan(shapes).any((1, 2))]
        ref = shapes[0]
        for _ in range(iters):
            aligned = np.stack([_align(s, ref) for s in shapes])
            ref = aligned.mean(0)
        flat = aligned.reshape(len(aligned), -1)
        cov = np.cov(flat, rowvar=False)
        cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov)) + 1e-3 * np.eye(cov.shape[0])
        return cls(ref, cov)

    def state(self):
        return {"mean": self.mean.tolist(), "cov": self.cov.tolist()}

    @classmethod
    def from_state(cls, st):
        return cls(st["mean"], st["cov"])

    def residuals(self, shape):
        """(K,) Mahalanobis distance of each landmark given the other K-1."""
        x = _align(np.asarray(shape, dtype=np.float64), self.mean).reshape(-1)
        out = []
        for a, o, gain, cinv in self.cond:
            r = x[a] - (self.mu[a] + gain @ (x[o] - self.mu[o]))
            out.append(float(np.sqrt(r @ cinv @ r)))
        return np.array(out)


# ---------------------------------------------------------------- combining
SIGNALS = ("spread", "conf", "shape")


def features(pred, shape_model):
    """(N,K,3) features: oscillation spread, policy confidence, shape residual."""
    shape = np.stack([shape_model.residuals(p) for p in pred["pred"]])
    return np.stack([pred["spread"], pred["conf"], shape], -1)


class FlagScorer:
    """z-normalise the chosen signals, then logistic regression for P(error > threshold)."""

    def __init__(self, signals=SIGNALS, threshold_mm=2.0):
        self.cols = [SIGNALS.index(s) for s in signals]
        self.threshold = threshold_mm

    def fit(self, feats, err):
        x, y = self._x(feats, fit=True), err.reshape(-1)
        keep = ~np.isnan(y)
        labels = y[keep] > self.threshold
        if labels.all() or not labels.any():
            # Validation errors are all on one side of the threshold (e.g. an untrained model):
            # fall back to an unweighted sum of z-scores, with confidence counting against.
            print(f"warning: all validation errors on one side of {self.threshold} mm; using z-score sum")
            self.lr = None
        else:
            self.lr = LogisticRegression().fit(x[keep], labels)
        return self

    def score(self, feats):
        x = self._x(feats)
        if self.lr is None:
            sign = np.array([-1.0 if SIGNALS[c] == "conf" else 1.0 for c in self.cols])
            return (x * sign).mean(1).reshape(feats.shape[:2])
        return self.lr.predict_proba(x)[:, 1].reshape(feats.shape[:2])

    def _x(self, feats, fit=False):
        x = np.nan_to_num(feats[..., self.cols].reshape(-1, len(self.cols)))
        if fit:
            self.mu, self.sd = x.mean(0), x.std(0) + 1e-8
        return (x - self.mu) / self.sd


# ---------------------------------------------------------------- metrics
def flag_metrics(score, err, budget=0.10, disagreement=None):
    s, e = score.reshape(-1), err.reshape(-1)
    keep = ~np.isnan(e)
    s, e = s[keep], e[keep]
    out = {}
    for t in (2.0, 4.0):
        y = e > t
        if 0 < y.sum() < len(y):
            out[f"AUROC_err>{t}mm"] = float(roc_auc_score(y, s))
            out[f"AUPRC_err>{t}mm"] = float(average_precision_score(y, s))
        else:
            out[f"AUROC_err>{t}mm"] = out[f"AUPRC_err>{t}mm"] = None
    n_flag = int(round(budget * len(s)))
    flagged = np.zeros(len(s), bool)
    flagged[np.argsort(-s)[:n_flag]] = True
    big = e > 4.0
    out[f"catch_rate_err>4mm_at_{int(budget * 100)}%"] = float(flagged[big].mean()) if big.any() else None
    out["MRE_unflagged_mm"] = float(e[~flagged].mean())

    order = np.argsort(-s)
    curve = []
    for frac in np.linspace(0, 0.5, 11):
        rest = order[int(round(frac * len(s))):]
        curve.append({"flagged_%": round(frac * 100), "MRE_unflagged_mm": float(e[rest].mean())})
    out["risk_coverage"] = curve
    if disagreement is not None:
        d = disagreement.reshape(-1)[keep]
        ok = ~np.isnan(d)
        out["spearman_score_vs_doctor_disagreement"] = float(spearmanr(s[ok], d[ok]).correlation)
    return out
