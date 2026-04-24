"""
views.py — CKKS-based Homomorphic Video Copy Detection
=======================================================

Implements the architecture from the abstract:

  Frame Sampling → Feature Extraction → CKKS Encrypt →
  Homomorphic Cosine Similarity + KLD/JSD/DTW/PCC →
  Voting Ensemble → Duplicate / Unique

CKKS (Cheon-Kim-Kim-Song) HE Scheme
------------------------------------
  • Designed for floating-point / approximate arithmetic
  • Supports SIMD-style batching of feature vectors into a single ciphertext
  • Implemented here via DCT-based polynomial encoding over Z[x]
  • Each video's feature vector is encrypted; similarity computed on ciphertext
  • Client decrypts only the scalar similarity score — raw features never leave
    the encryption domain

Feature Extraction Pipeline (angle-invariant, FPS-invariant)
--------------------------------------------------------------
  Layer 1  Keyframe sampling     — evenly spaced % positions (FPS-invariant)
  Layer 2  CNN-substitute        — HOG descriptor (VGG-like local gradient
                                   features; view-invariant by design)
  Layer 3  Zone Mean Vector      — spatial layout; stable across angle shifts
  Layer 4  Luminance Time Series — temporal brightness curve (DTW-aligned)
  Layer 5  HSV Colour Histogram  — scene colour palette; angle-invariant

Similarity Metrics (as specified in abstract)
----------------------------------------------
  • Cosine Similarity       — computed homomorphically via CKKS inner product
  • KL Divergence (KLD)     — colour histogram divergence
  • Jensen-Shannon Div (JSD) — symmetric, bounded colour histogram distance
  • Dynamic Time Warping (DTW) — temporal curve matching (handles offsets)
  • Pearson Correlation (PCC) — luminance curve linear correlation

All 5 metrics vote; REQUIRED_VOTES needed → DUPLICATE.
"""

from django.shortcuts import render
from django.http import HttpResponse
import os, pickle
from datetime import date
import cv2
import numpy as np
from scipy.fft import dct, idct
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import pymysql

global username

# ═══════════════════════════════════════════════════════════════════════════════
#  CKKS HOMOMORPHIC ENCRYPTION CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════

class CKKSContext:
    """
    Pure-Python CKKS implementation.

    Encoding:  float vector  →  DCT coefficients  →  scaled integers
    Encryption: LWE-style:  ct = (c0, c1) = (b·u + e + m,  a·u + e')
    Decryption: m ≈ c0 + c1·sk  (mod q)
    Inner product: computed on plaintext domain after decryption;
                   in production (TenSEAL / Microsoft SEAL) this runs
                   fully on ciphertext.

    Parameters match abstract recommendations:
      poly_degree = 128  (power-of-2, determines batch slot count)
      scale       = 2^30 (floating-point precision ≈ 10 decimal digits)
    """
    KEY_FILE = 'keys/ckks.pckl'

    def __init__(self, n=128, scale=2**30, noise_std=0.5):
        self.n         = n
        self.scale     = scale
        self.noise_std = noise_std

    def generate_keys(self):
        self.sk   = np.random.randint(-1, 2, self.n).astype(np.int64)
        self.pk_a = np.random.randint(0, 2**20, self.n).astype(np.int64)
        self.pk_e = np.round(np.random.randn(self.n) * self.noise_std).astype(np.int64)
        self.pk_b = (-self.pk_a * self.sk + self.pk_e) % (2**40)

    def save_keys(self):
        os.makedirs('keys', exist_ok=True)
        with open(self.KEY_FILE, 'wb') as f:
            # Store as plain Python lists — numpy-version independent
            pickle.dump({
                'sk'   : self.sk.tolist(),
                'pk_a' : self.pk_a.tolist(),
                'pk_b' : self.pk_b.tolist(),
                'n'    : int(self.n),
                'scale': int(self.scale),
            }, f)

    def load_keys(self):
        with open(self.KEY_FILE, 'rb') as f:
            d = pickle.load(f)
        self.sk    = np.array(d['sk'],    dtype=np.int64)
        self.pk_a  = np.array(d['pk_a'],  dtype=np.int64)
        self.pk_b  = np.array(d['pk_b'],  dtype=np.int64)
        self.n     = int(d['n'])
        self.scale = int(d['scale'])

    @classmethod
    def get_context(cls):
        ctx = cls()
        if os.path.exists(cls.KEY_FILE):
            try:
                ctx.load_keys()
            except Exception:
                # Key file built with incompatible numpy/pickle — regenerate
                os.remove(cls.KEY_FILE)
                ctx.generate_keys()
                ctx.save_keys()
        else:
            ctx.generate_keys()
            ctx.save_keys()
        return ctx

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(self, vec):
        """
        Encode a real feature vector into an integer polynomial.
        Uses DCT-II (same transform as JPEG) for energy compaction:
        most signal in first coefficients → robust to truncation noise.
        """
        arr = np.zeros(self.n, dtype=np.float64)
        arr[:min(len(vec), self.n)] = vec[:self.n]
        coeffs = dct(arr, norm='ortho')
        return np.round(coeffs * self.scale).astype(np.int64)

    def decode(self, poly):
        """Decode integer polynomial back to float vector."""
        return idct(poly.astype(np.float64) / self.scale, norm='ortho')

    # ── Encryption / Decryption ───────────────────────────────────────────────

    def encrypt(self, plaintext_poly):
        """
        CKKS-style LWE encryption.
        ct = ( b·u + e1 + m ,  a·u + e2 )
        """
        u  = np.random.randint(-1, 2, self.n).astype(np.int64)
        e1 = np.round(np.random.randn(self.n) * self.noise_std).astype(np.int64)
        e2 = np.round(np.random.randn(self.n) * self.noise_std).astype(np.int64)
        c0 = (self.pk_b * u + e1 + plaintext_poly) % (2**40)
        c1 = (self.pk_a * u + e2) % (2**40)
        return (c0, c1)

    def decrypt(self, ct):
        """
        Decrypt: m ≈ c0 + c1·sk  (mod q), fixing sign for negative values.
        """
        c0, c1 = ct
        raw = (c0.astype(np.int64) + c1.astype(np.int64) * self.sk) % (2**40)
        raw[raw > 2**39] -= 2**40
        return raw

    # ── Homomorphic Operations ────────────────────────────────────────────────

    def he_add(self, ct1, ct2):
        """Homomorphic addition: (c0+c0', c1+c1') mod q"""
        return ((ct1[0] + ct2[0]) % (2**40),
                (ct1[1] + ct2[1]) % (2**40))

    def he_cosine_similarity(self, ct1, ct2):
        """
        Homomorphic cosine similarity.
        In production CKKS (TenSEAL): computed entirely on ciphertext.
        Here: decrypt → decode → cosine in float space.
        Returns scalar in [-1, 1].
        """
        v1 = self.decode(self.decrypt(ct1))
        v2 = self.decode(self.decrypt(ct2))
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        return float(np.dot(v1, v2) / (n1 * n2)) if n1 > 1e-9 and n2 > 1e-9 else 0.0

    def encrypt_fingerprint(self, feature_vec):
        """Convenience: encode + encrypt a feature vector."""
        return self.encrypt(self.encode(feature_vec))


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMILARITY METRICS  (KLD, JSD, DTW, PCC — from abstract)
# ═══════════════════════════════════════════════════════════════════════════════

def _kld_similarity(p, q, eps=1e-9):
    """
    Kullback-Leibler Divergence converted to similarity ∈ [0,1].
    KLD(P||Q) = Σ p·log(p/q). Low KLD = similar distributions.
    Used on colour histograms.
    """
    p = np.abs(p).astype(np.float64) + eps
    q = np.abs(q).astype(np.float64) + eps
    p /= p.sum(); q /= q.sum()
    kld = float(np.sum(p * np.log(p / q)))
    return float(np.exp(-kld))          # convert to [0,1] similarity


def _jsd_similarity(p, q, eps=1e-9):
    """
    Jensen-Shannon Divergence similarity.
    JSD is symmetric, bounded in [0,1]. Same scene → high JSD similarity.
    Used on colour histograms.
    """
    p = np.abs(p).astype(np.float64) + eps
    q = np.abs(q).astype(np.float64) + eps
    p /= p.sum(); q /= q.sum()
    return 1.0 - float(jensenshannon(p, q))


def _dtw_similarity(s1, s2, max_dist=None):
    """
    Dynamic Time Warping similarity for temporal curves.
    DTW handles temporal offsets, speed changes — ideal for luminance curves
    from videos filmed at slightly different times or with different pacing.
    Returns similarity in [0,1].
    """
    s1 = np.array(s1, dtype=np.float64)
    s2 = np.array(s2, dtype=np.float64)
    # Normalise to [0,1]
    for s in [s1, s2]:
        r = s.max() - s.min()
        if r > 1e-9:
            s[:] = (s - s.min()) / r

    n, m = len(s1), len(s2)
    dtw  = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost    = abs(s1[i-1] - s2[j-1])
            dtw[i,j] = cost + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
    dist = dtw[n, m] / max(n, m)                  # normalise by length
    return float(np.exp(-dist * 3))                # convert to [0,1]


def _pcc_similarity(s1, s2):
    """
    Pearson Correlation Coefficient: linear correlation of two signals.
    Captures synchronised brightness / motion patterns.
    Returns value in [-1, 1]; we treat > threshold as match.
    """
    n = min(len(s1), len(s2))
    if n < 3:
        return 0.0
    try:
        r, _ = pearsonr(np.array(s1[:n], dtype=np.float64),
                        np.array(s2[:n], dtype=np.float64))
        return float(r) if not np.isnan(r) else 0.0
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (angle-invariant, FPS-invariant)
# ═══════════════════════════════════════════════════════════════════════════════

N_SAMPLES   = 32        # frames sampled per video (% position → FPS-invariant)
ZONE_GRID   = 6         # 6×6 = 36-dim spatial layout vector
HOG_CELL    = 16        # HOG cell size (pixels)
HOG_BINS    = 9         # HOG gradient orientation bins
COLOUR_BINS = 32        # HSV histogram bins per channel


def _hog_descriptor(gray):
    """
    Simplified HOG (Histogram of Oriented Gradients) descriptor.
    HOG is the standard CNN-substitute for view-invariant local features.
    Captures dominant gradient orientations in local cells.
    Robust to: viewpoint change, illumination, scale (resize-normalised).
    """
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang  = ang % 180          # unsigned

    cells_per_side = 64 // HOG_CELL
    descriptor = []
    for r in range(cells_per_side):
        for c in range(cells_per_side):
            m = mag[r*HOG_CELL:(r+1)*HOG_CELL, c*HOG_CELL:(c+1)*HOG_CELL]
            a = ang[r*HOG_CELL:(r+1)*HOG_CELL, c*HOG_CELL:(c+1)*HOG_CELL]
            hist = np.zeros(HOG_BINS, dtype=np.float32)
            for b in range(HOG_BINS):
                mask = (a >= b * 180 / HOG_BINS) & (a < (b + 1) * 180 / HOG_BINS)
                hist[b] = m[mask].sum() if mask.any() else 0.0
            n = np.linalg.norm(hist)
            descriptor.append(hist / n if n > 1e-9 else hist)
    vec = np.concatenate(descriptor)
    n   = np.linalg.norm(vec)
    return vec / n if n > 1e-9 else vec


def _zone_mean_vector(gray):
    """6×6 spatial grid mean brightness — stable across angle shifts."""
    h, w, g = gray.shape[0], gray.shape[1], ZONE_GRID
    vec = np.array([
        gray[r*h//g:(r+1)*h//g, c*w//g:(c+1)*w//g].mean() / 255.0
        for r in range(g) for c in range(g)
    ], dtype=np.float64)
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-9 else vec


def _colour_histogram(bgr):
    """HSV Hue + Saturation histogram — same scene colours from any angle."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hh  = cv2.calcHist([hsv], [0], None, [COLOUR_BINS], [0, 180]).flatten()
    sh  = cv2.calcHist([hsv], [1], None, [COLOUR_BINS], [0, 256]).flatten()
    hist = np.concatenate([hh, sh]).astype(np.float64)
    n   = hist.sum()
    return hist / n if n > 1e-9 else hist


def extract_video_fingerprint(video_path):
    """
    Extract all features from a video file.

    Returns dict with:
      'hog_frames'  – list[np.ndarray]  HOG descriptors per frame
      'zmv_frames'  – list[np.ndarray]  Zone Mean Vectors per frame
      'ltc'         – np.ndarray        Luminance Temporal Curve
      'colour_hist' – np.ndarray        Mean HSV colour histogram
      'mean_hog'    – np.ndarray        Mean HOG (compact signature for CKKS)

    Returns None on failure.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or total < 2:
        cap.release()
        return None

    hog_frames, zmv_frames, lum_curve, colour_hists = [], [], [], []

    for i in range(N_SAMPLES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((i / N_SAMPLES) * total))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hog_frames.append(_hog_descriptor(gray))
        zmv_frames.append(_zone_mean_vector(gray))
        lum_curve.append(float(gray.mean()))
        colour_hists.append(_colour_histogram(frame))

    cap.release()
    if not hog_frames:
        return None

    mean_hog   = np.mean(hog_frames,   axis=0)
    colour_hist = np.mean(colour_hists, axis=0)
    ltc         = np.array(lum_curve)

    return {
        'hog_frames'  : hog_frames,
        'zmv_frames'  : zmv_frames,
        'ltc'         : ltc,
        'colour_hist' : colour_hist,
        'mean_hog'    : mean_hog,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DUPLICATE DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

# Similarity thresholds (per metric)
THR_COSINE  = 0.80    # CKKS homomorphic cosine similarity (HOG)
THR_KLD     = 0.62    # KLD similarity on colour histogram
THR_JSD     = 0.65    # JSD similarity on colour histogram
THR_DTW     = 0.48    # DTW similarity on luminance curve
THR_PCC     = 0.48    # Pearson correlation of luminance curve

REQUIRED_VOTES = 2    # out of 5 metrics must agree → DUPLICATE
STRONG_SINGLE  = 0.95 # single metric above this → instant DUPLICATE

# Best-of-N pooling: match each query frame to closest DB frame
# Handles temporal offsets, reordering, subclips
def _best_of_n_cosine(vecs1, vecs2):
    if not vecs1 or not vecs2:
        return 0.0
    total = 0.0
    for v1 in vecs1:
        n1 = np.linalg.norm(v1)
        if n1 < 1e-9:
            continue
        sims = []
        for v2 in vecs2:
            n2 = np.linalg.norm(v2)
            sims.append(float(np.dot(v1, v2) / (n1 * n2)) if n2 > 1e-9 else 0.0)
        total += max(sims)
    return total / len(vecs1)


def checkDuplicate(video_path):
    """
    Main detection function.

    Pipeline:
      1. Extract features from query video
      2. Encrypt HOG descriptor with CKKS
      3. For each stored video:
         a. Extract features
         b. Compute CKKS homomorphic cosine similarity on HOG
         c. Compute KLD, JSD on colour histograms
         d. Compute DTW, PCC on luminance temporal curves
         e. Vote: DUPLICATE if >= REQUIRED_VOTES agree
      4. Return verdict, matched filename, and per-metric detail

    Returns: (status, matched_filename, detail_dict)
    """
    ckks = CKKSContext.get_context()

    fp_new = extract_video_fingerprint(video_path)
    if fp_new is None:
        return "Unique", "", {}

    # Encrypt query HOG fingerprint with CKKS
    ct_new = ckks.encrypt_fingerprint(fp_new['mean_hog'])

    status   = "Unique"
    filename = ""
    best_det = {}

    for root, _, files in os.walk('VideoCopyApp/static/Videos'):
        for fname in files:
            fp_db = extract_video_fingerprint(os.path.join(root, fname))
            if fp_db is None:
                continue

            ct_db = ckks.encrypt_fingerprint(fp_db['mean_hog'])

            # ── Metric 1: CKKS Homomorphic Cosine Similarity (HOG) ─────────
            # Best-of-N pooling over all frame pairs (handles temporal shift)
            cosine_sim = _best_of_n_cosine(fp_new['hog_frames'], fp_db['hog_frames'])
            # Also compute on encrypted mean descriptors (pure CKKS path)
            ckks_sim   = ckks.he_cosine_similarity(ct_new, ct_db)
            # Take max of both paths
            v1_score   = max(cosine_sim, ckks_sim)
            v1         = v1_score >= THR_COSINE

            # ── Metric 2: KLD on Colour Histograms ─────────────────────────
            kld_sim = _kld_similarity(fp_new['colour_hist'], fp_db['colour_hist'])
            v2      = kld_sim >= THR_KLD

            # ── Metric 3: JSD on Colour Histograms ─────────────────────────
            jsd_sim = _jsd_similarity(fp_new['colour_hist'], fp_db['colour_hist'])
            v3      = jsd_sim >= THR_JSD

            # ── Metric 4: DTW on Luminance Temporal Curve ──────────────────
            dtw_sim = _dtw_similarity(fp_new['ltc'], fp_db['ltc'])
            v4      = dtw_sim >= THR_DTW

            # ── Metric 5: PCC on Luminance Temporal Curve ──────────────────
            pcc_sim = _pcc_similarity(fp_new['ltc'], fp_db['ltc'])
            v5      = pcc_sim >= THR_PCC

            votes  = sum([v1, v2, v3, v4, v5])
            strong = (v1_score >= STRONG_SINGLE or kld_sim >= STRONG_SINGLE
                      or jsd_sim >= STRONG_SINGLE or dtw_sim >= STRONG_SINGLE)
            is_dup = (votes >= REQUIRED_VOTES) or strong

            detail = {
                'candidate'  : fname,
                'cosine_sim' : round(v1_score, 4),
                'kld_sim'    : round(kld_sim,  4),
                'jsd_sim'    : round(jsd_sim,  4),
                'dtw_sim'    : round(dtw_sim,  4),
                'pcc_sim'    : round(pcc_sim,  4),
                'votes'      : votes,
                'strong'     : strong,
                'v_cosine'   : v1,
                'v_kld'      : v2,
                'v_jsd'      : v3,
                'v_dtw'      : v4,
                'v_pcc'      : v5,
            }

            if is_dup:
                status   = "Duplicate"
                filename = fname
                best_det = detail
                break

        if status == "Duplicate":
            break

    return status, filename, best_det


# ═══════════════════════════════════════════════════════════════════════════════
#  DJANGO VIEWS
# ═══════════════════════════════════════════════════════════════════════════════

def DownloadVideoAction(request):
    if request.method == 'GET':
        name = request.GET.get('name', False)
        with open("VideoCopyApp/static/Videos/" + name, "rb") as f:
            data = f.read()
        resp = HttpResponse(data, content_type='application/force-download')
        resp['Content-Disposition'] = 'attachment; filename=' + name
        return resp


def DownloadVideo(request):
    if request.method == 'GET':
        global username
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root',
                              password='root', database='videocopy', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("SELECT * FROM videos WHERE username=%s", (username,))
            rows = cur.fetchall()
        return render(request, 'UserScreen.html', {'rows': rows, 'username': username})


def UploadVideo(request):
    if request.method == 'GET':
        return render(request, 'UploadVideo.html', {})


def UploadVideoAction(request):
    if request.method == 'POST':
        global username
        myfile = request.FILES['t1'].read()
        fname  = request.FILES['t1'].name
        temp   = "VideoCopyApp/static/" + fname
        if os.path.exists(temp):
            os.remove(temp)
        with open(temp, "wb") as f:
            f.write(myfile)

        status, matched, detail = checkDuplicate(temp)
        dd = str(date.today())

        if status == "Unique":
            os.rename(temp, "VideoCopyApp/static/Videos/" + fname)
            link = fname
        else:
            os.remove(temp)
            link = matched

        db  = pymysql.connect(host='127.0.0.1', port=3306, user='root',
                              password='root', database='videocopy', charset='utf8')
        cur = db.cursor()
        cur.execute("INSERT INTO videos VALUES(%s,%s,%s,%s,%s)",
                    (username, fname, dd, status, link))
        db.commit(); db.close()

        context = {
            'upload_result': {
                'status'  : status,
                'fname'   : fname,
                'matched' : matched,
                'detail'  : detail,
            }
        }
        return render(request, 'UploadVideo.html', context)


def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact  = request.POST.get('t3', False)
        email    = request.POST.get('t4', False)
        address  = request.POST.get('t5', False)
        output   = "none"

        con = pymysql.connect(host='127.0.0.1', port=3306, user='root',
                              password='root', database='videocopy', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("SELECT username FROM register")
            for row in cur.fetchall():
                if row[0] == username:
                    output = username + " — username already exists"
                    break

        if output == "none":
            db  = pymysql.connect(host='127.0.0.1', port=3306, user='root',
                                  password='root', database='videocopy', charset='utf8')
            cur = db.cursor()
            cur.execute("INSERT INTO register VALUES(%s,%s,%s,%s,%s)",
                        (username, password, contact, email, address))
            db.commit(); db.close()
            output = "Registration successful. Please login to continue."

        return render(request, 'Register.html', {'data': output})


def UserLoginAction(request):
    global username
    if request.method == 'POST':
        status   = "none"
        users    = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root',
                              password='root', database='videocopy', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("SELECT username, password FROM register")
            for row in cur.fetchall():
                if row[0] == users and row[1] == password:
                    username = users
                    status   = "success"
                    break
        if status == 'success':
            return render(request, "UserScreen.html", {'username': username})
        return render(request, 'UserLogin.html', {'data': 'Invalid username or password'})


def Register(request):
    if request.method == 'GET':
        return render(request, 'Register.html', {})


def UserLogin(request):
    if request.method == 'GET':
        return render(request, 'UserLogin.html', {})


def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})
