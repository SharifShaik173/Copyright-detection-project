"""
test.py — Standalone CKKS Video Copy Detection Test
====================================================
Usage:
    python test.py <query_video> [<db_folder>]
    python test.py                           # runs synthetic self-test

Mirrors views.py exactly. Prints per-metric similarity scores + verdict.
"""
import os, sys, pickle
import cv2
import numpy as np
from scipy.fft import dct, idct
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon

# ── Parameters (keep in sync with views.py) ──────────────────────────────────
N_SAMPLES      = 32
ZONE_GRID      = 6
HOG_CELL       = 16
HOG_BINS       = 9
COLOUR_BINS    = 32
THR_COSINE     = 0.80
THR_KLD        = 0.62
THR_JSD        = 0.65
THR_DTW        = 0.48
THR_PCC        = 0.48
REQUIRED_VOTES = 2
STRONG_SINGLE  = 0.95

# ── CKKS Context ─────────────────────────────────────────────────────────────
class CKKSContext:
    KEY_FILE = 'keys/ckks.pckl'

    def __init__(self, n=128, scale=2**30, noise_std=0.5):
        self.n = n; self.scale = scale; self.noise_std = noise_std

    def generate_keys(self):
        self.sk   = np.random.randint(-1, 2, self.n).astype(np.int64)
        self.pk_a = np.random.randint(0, 2**20, self.n).astype(np.int64)
        self.pk_e = np.round(np.random.randn(self.n)*self.noise_std).astype(np.int64)
        self.pk_b = (-self.pk_a*self.sk + self.pk_e) % (2**40)

    def save_keys(self):
        os.makedirs('keys', exist_ok=True)
        with open(self.KEY_FILE,'wb') as f:
            pickle.dump({'sk':self.sk.tolist(),'pk_a':self.pk_a.tolist(),
                         'pk_b':self.pk_b.tolist(),'n':int(self.n),
                         'scale':int(self.scale)}, f)

    def load_keys(self):
        with open(self.KEY_FILE,'rb') as f: d=pickle.load(f)
        self.sk=np.array(d['sk'],dtype=np.int64); self.pk_a=np.array(d['pk_a'],dtype=np.int64)
        self.pk_b=np.array(d['pk_b'],dtype=np.int64); self.n=int(d['n']); self.scale=int(d['scale'])

    @classmethod
    def get_context(cls):
        ctx=cls()
        if os.path.exists(cls.KEY_FILE):
            try:
                ctx.load_keys()
            except Exception:
                os.remove(cls.KEY_FILE)
                ctx.generate_keys(); ctx.save_keys()
        else:
            ctx.generate_keys(); ctx.save_keys()
        return ctx

    def encode(self, vec):
        arr=np.zeros(self.n,dtype=np.float64); arr[:min(len(vec),self.n)]=vec[:self.n]
        return np.round(dct(arr,norm='ortho')*self.scale).astype(np.int64)

    def decode(self, poly):
        return idct(poly.astype(np.float64)/self.scale, norm='ortho')

    def encrypt(self, p):
        u=np.random.randint(-1,2,self.n).astype(np.int64)
        e1=np.round(np.random.randn(self.n)*self.noise_std).astype(np.int64)
        e2=np.round(np.random.randn(self.n)*self.noise_std).astype(np.int64)
        c0=(self.pk_b*u+e1+p)%(2**40); c1=(self.pk_a*u+e2)%(2**40)
        return (c0,c1)

    def decrypt(self, ct):
        c0,c1=ct; raw=(c0.astype(np.int64)+c1.astype(np.int64)*self.sk)%(2**40)
        raw[raw>2**39]-=2**40; return raw

    def he_cosine_similarity(self, ct1, ct2):
        v1=self.decode(self.decrypt(ct1)); v2=self.decode(self.decrypt(ct2))
        n1=np.linalg.norm(v1); n2=np.linalg.norm(v2)
        return float(np.dot(v1,v2)/(n1*n2)) if n1>1e-9 and n2>1e-9 else 0.0

    def encrypt_fingerprint(self, vec): return self.encrypt(self.encode(vec))

# ── Metrics ──────────────────────────────────────────────────────────────────
def kld_sim(p,q,e=1e-9):
    p=np.abs(p)+e; q=np.abs(q)+e; p/=p.sum(); q/=q.sum()
    return float(np.exp(-np.sum(p*np.log(p/q))))

def jsd_sim(p,q,e=1e-9):
    p=np.abs(p)+e; q=np.abs(q)+e; p/=p.sum(); q/=q.sum()
    return 1.0-float(jensenshannon(p,q))

def dtw_sim(s1,s2):
    s1=np.array(s1,dtype=np.float64); s2=np.array(s2,dtype=np.float64)
    for s in [s1,s2]:
        r=s.max()-s.min()
        if r>1e-9: s[:]=(s-s.min())/r
    n,m=len(s1),len(s2); dtw=np.full((n+1,m+1),np.inf); dtw[0,0]=0
    for i in range(1,n+1):
        for j in range(1,m+1):
            dtw[i,j]=abs(s1[i-1]-s2[j-1])+min(dtw[i-1,j],dtw[i,j-1],dtw[i-1,j-1])
    return float(np.exp(-dtw[n,m]/max(n,m)*3))

def pcc_sim(s1,s2):
    n=min(len(s1),len(s2))
    if n<3: return 0.0
    try: r,_=pearsonr(np.array(s1[:n]),np.array(s2[:n])); return float(r) if not np.isnan(r) else 0.0
    except: return 0.0

# ── Feature helpers ──────────────────────────────────────────────────────────
def hog_desc(gray):
    gray=cv2.resize(gray,(64,64),interpolation=cv2.INTER_AREA)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    mag,ang=cv2.cartToPolar(gx,gy,angleInDegrees=True); ang=ang%180
    cps=64//HOG_CELL; desc=[]
    for r in range(cps):
        for c in range(cps):
            m=mag[r*HOG_CELL:(r+1)*HOG_CELL,c*HOG_CELL:(c+1)*HOG_CELL]
            a=ang[r*HOG_CELL:(r+1)*HOG_CELL,c*HOG_CELL:(c+1)*HOG_CELL]
            h=np.array([m[(a>=b*20)&(a<(b+1)*20)].sum() for b in range(HOG_BINS)],dtype=np.float32)
            n=np.linalg.norm(h); desc.append(h/n if n>1e-9 else h)
    v=np.concatenate(desc); n=np.linalg.norm(v); return v/n if n>1e-9 else v

def zmv(gray):
    h,w,g=gray.shape[0],gray.shape[1],ZONE_GRID
    v=np.array([gray[r*h//g:(r+1)*h//g,c*w//g:(c+1)*w//g].mean()/255.0 for r in range(g) for c in range(g)])
    n=np.linalg.norm(v); return v/n if n>1e-9 else v

def chist(bgr):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    hh=cv2.calcHist([hsv],[0],None,[COLOUR_BINS],[0,180]).flatten()
    sh=cv2.calcHist([hsv],[1],None,[COLOUR_BINS],[0,256]).flatten()
    h=np.concatenate([hh,sh]); n=h.sum(); return h/n if n>1e-9 else h

def fingerprint(path):
    cap=cv2.VideoCapture(path); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or total<2: cap.release(); return None
    hogs,lums,cols=[],[],[]
    for i in range(N_SAMPLES):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int((i/N_SAMPLES)*total))
        ret,frame=cap.read()
        if not ret: continue
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        hogs.append(hog_desc(gray)); lums.append(float(gray.mean())); cols.append(chist(frame))
    cap.release()
    if not hogs: return None
    return {'hog_frames':hogs,'mean_hog':np.mean(hogs,axis=0),
            'ltc':np.array(lums),'colour_hist':np.mean(cols,axis=0)}

def bon_cosine(v1s,v2s):
    if not v1s or not v2s: return 0.0
    total=0.0
    for v1 in v1s:
        n1=np.linalg.norm(v1)
        if n1<1e-9: continue
        total+=max(float(np.dot(v1,v2)/(n1*(np.linalg.norm(v2)+1e-9))) for v2 in v2s)
    return total/len(v1s)

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    ckks = CKKSContext.get_context()

    if len(sys.argv) < 2:
        print("\n[SELF-TEST MODE — no video files needed]\n")
        # Synthetic validation
        np.random.seed(7)
        v1=np.random.rand(144); v1/=np.linalg.norm(v1)
        v_same=v1+np.random.randn(144)*0.03; v_same/=np.linalg.norm(v_same)
        v_diff=np.random.rand(144); v_diff/=np.linalg.norm(v_diff)
        ltc1=np.cumsum(np.random.randn(32))*2+128
        ltc_same=ltc1+np.random.randn(32)*3
        ltc_diff=np.cumsum(np.random.randn(32))*2+90
        ch1=np.abs(np.random.randn(64))+0.1; ch1/=ch1.sum()
        ch_same=ch1*(1+np.random.randn(64)*0.03); ch_same/=ch_same.sum()
        ch_diff=np.abs(np.random.randn(64))+0.1; ch_diff/=ch_diff.sum()

        for label,v,lt,ch in [("SAME SCENE (diff angle)",v_same,ltc_same,ch_same),
                               ("DIFFERENT SCENE       ",v_diff,ltc_diff,ch_diff)]:
            ct1=ckks.encrypt_fingerprint(v1); ct2=ckks.encrypt_fingerprint(v)
            cs=ckks.he_cosine_similarity(ct1,ct2)
            k=kld_sim(ch1,ch); j=jsd_sim(ch1,ch)
            d=dtw_sim(ltc1,lt); p=pcc_sim(ltc1,lt)
            vv=sum([cs>=THR_COSINE,k>=THR_KLD,j>=THR_JSD,d>=THR_DTW,p>=THR_PCC])
            strong=cs>=STRONG_SINGLE or k>=STRONG_SINGLE or j>=STRONG_SINGLE
            res="DUPLICATE ✓" if (vv>=REQUIRED_VOTES or strong) else "UNIQUE ✓" if label.startswith("DIFFERENT") else "unique ✗ MISS"
            print(f"  {label}")
            print(f"    Cosine={cs:.3f}  KLD={k:.3f}  JSD={j:.3f}  DTW={d:.3f}  PCC={p:.3f}")
            print(f"    Votes={vv}/5  Strong={strong}  → {res}\n")
        sys.exit(0)

    query  = sys.argv[1]
    db_dir = sys.argv[2] if len(sys.argv)>2 else 'Videos'
    print(f"\n{'='*70}")
    print(f"  Query : {query}")
    print(f"  DB    : {db_dir}")
    print(f"{'='*70}")
    print(f"  {'FILE':28s}  COS    KLD    JSD    DTW    PCC   VOTES  RESULT")
    print(f"  {'-'*28}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*9}")

    fp_q = fingerprint(query)
    if fp_q is None: print("ERROR: Could not read query video."); sys.exit(1)
    ct_q = ckks.encrypt_fingerprint(fp_q['mean_hog'])
    verdict="UNIQUE"; match=""

    for root,_,files in os.walk(db_dir):
        for fname in files:
            fp = fingerprint(os.path.join(root,fname))
            if fp is None: continue
            ct=ckks.encrypt_fingerprint(fp['mean_hog'])
            cs=max(bon_cosine(fp_q['hog_frames'],fp['hog_frames']),
                   ckks.he_cosine_similarity(ct_q,ct))
            k=kld_sim(fp_q['colour_hist'],fp['colour_hist'])
            j=jsd_sim(fp_q['colour_hist'],fp['colour_hist'])
            d=dtw_sim(fp_q['ltc'],fp['ltc'])
            p=pcc_sim(fp_q['ltc'],fp['ltc'])
            vv=sum([cs>=THR_COSINE,k>=THR_KLD,j>=THR_JSD,d>=THR_DTW,p>=THR_PCC])
            strong=cs>=STRONG_SINGLE or k>=STRONG_SINGLE or j>=STRONG_SINGLE
            is_dup=vv>=REQUIRED_VOTES or strong
            tag="DUPLICATE ✗" if is_dup else "unique    ✓"
            print(f"  {fname:28s}  {cs:.3f}  {k:.3f}  {j:.3f}  {d:.3f}  {p:.3f}  {vv}/5   {tag}")
            if is_dup and verdict=="UNIQUE": verdict="DUPLICATE"; match=fname

    print(f"\n{'─'*70}")
    print(f"  VERDICT: {verdict}"+(f"  →  matches '{match}'" if match else ""))
    print(f"{'─'*70}\n")
