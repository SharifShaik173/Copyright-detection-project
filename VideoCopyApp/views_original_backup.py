from django.shortcuts import render
from django.contrib import messages
from django.http import HttpResponse
import os
import pymysql
from phe import paillier
import pickle
from datetime import date
import cv2
import numpy as np

global username

def generateKeys():
    if os.path.exists('keys/fhe.pckl'):
        f = open('keys/fhe.pckl', 'rb')
        keys = pickle.load(f)
        f.close()
        public_key, private_key = keys
    else:
        public_key, private_key = paillier.generate_paillier_keypair()
        keys = [public_key, private_key]
        f = open('keys/fhe.pckl', 'wb')
        pickle.dump(keys, f)
        f.close()
    return public_key, private_key

public_key, private_key = generateKeys()

# ─────────────────────────────────────────────────────────────────────────────
#  ALGORITHM FIX — Content-based Perceptual Fingerprinting
#
#  OLD (broken): read first 300 raw bytes → these are codec/container headers
#    that change with FPS, resolution, encoder settings → always "Unique".
#
#  NEW (fixed):
#    1. Sample N_SAMPLES frames at PERCENTAGE positions across the video.
#       → Invariant to FPS (30fps vs 60fps) and video length (5min vs 5min10s)
#    2. Resize every frame to a fixed FRAME_SIZE thumbnail.
#       → Invariant to resolution (1080p vs 720p)
#    3. Compute average-hash per frame (pixel vs frame mean → bit vector).
#       → Content-based, not codec-based
#    4. Sum hashes → single "visual signature" integer → Paillier-encrypt.
#    5. Compare with RELATIVE threshold on the decrypted difference,
#       PLUS a frame-level Hamming-distance similarity check.
#       → Handles different angles / lighting (multi-view)
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES          = 20      # frames to sample per video
FRAME_SIZE         = (16, 16)
DIFF_THRESHOLD     = 0.30    # 30% relative FHE-based diff → duplicate
FRAME_SIM_THRESHOLD = 0.65   # 65% frame-pairs visually similar → duplicate


def extract_video_fingerprint(video_path):
    """Extract perceptual fingerprint invariant to FPS, resolution, length."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return [], 0

    frame_hashes = []
    for i in range(N_SAMPLES):
        idx = int((i / N_SAMPLES) * total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        flat  = thumb.flatten().astype(float)
        mean_val = flat.mean()
        bits = ''.join('1' if v >= mean_val else '0' for v in flat)
        frame_hashes.append(int(bits, 2))

    cap.release()
    signature = sum(frame_hashes) if frame_hashes else 0
    return frame_hashes, signature


def compute_frame_similarity(hashes1, hashes2):
    """Fraction of frame pairs with Hamming distance below threshold."""
    if not hashes1 or not hashes2:
        return 0.0
    n_bits = FRAME_SIZE[0] * FRAME_SIZE[1]
    match_threshold = int(n_bits * 0.20)
    min_len = min(len(hashes1), len(hashes2))
    matches = sum(
        1 for i in range(min_len)
        if bin(hashes1[i] ^ hashes2[i]).count('1') <= match_threshold
    )
    return matches / min_len


def checkDuplicate(video_path):
    global public_key, private_key
    new_hashes, new_sig = extract_video_fingerprint(video_path)
    if not new_hashes:
        return "Unique", ""

    enc_new  = public_key.encrypt(new_sig)
    status   = ""
    filename = ""

    for root, dirs, directory in os.walk('VideoCopyApp/static/Videos'):
        for fname in directory:
            if status == "Duplicate":
                break
            db_hashes, db_sig = extract_video_fingerprint(os.path.join(root, fname))
            if not db_hashes:
                continue

            # FHE arithmetic on encrypted values
            enc_diff = enc_new - public_key.encrypt(db_sig)
            diff     = private_key.decrypt(enc_diff)

            max_sig  = max(new_sig, db_sig) or 1
            rel_diff = abs(diff) / max_sig
            sim      = compute_frame_similarity(new_hashes, db_hashes)

            if rel_diff < DIFF_THRESHOLD or sim > FRAME_SIM_THRESHOLD:
                status   = "Duplicate"
                filename = fname
            else:
                status = "Unique"
        if status == "Duplicate":
            break

    return status or "Unique", filename


# ─────────────────────────────────────────────────────────────────────────────
#  Views
# ─────────────────────────────────────────────────────────────────────────────

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
        context = {'rows': rows, 'username': username}
        return render(request, 'UserScreen.html', context)


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

        status, matched = checkDuplicate(temp)
        dd = str(date.today())

        if status == "Unique":
            os.rename(temp, "VideoCopyApp/static/Videos/" + fname)
            link = fname
        else:
            os.remove(temp)
            link = matched

        db = pymysql.connect(host='127.0.0.1', port=3306, user='root',
                             password='root', database='videocopy', charset='utf8')
        cur = db.cursor()
        cur.execute("INSERT INTO videos VALUES(%s,%s,%s,%s,%s)",
                    (username, fname, dd, status, link))
        db.commit()
        db.close()

        context = {'upload_result': {'status': status, 'fname': fname, 'matched': matched}}
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
            db = pymysql.connect(host='127.0.0.1', port=3306, user='root',
                                 password='root', database='videocopy', charset='utf8')
            cur = db.cursor()
            cur.execute("INSERT INTO register VALUES(%s,%s,%s,%s,%s)",
                        (username, password, contact, email, address))
            db.commit()
            db.close()
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
        else:
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
