import streamlit as st
import numpy as np
import cv2
from PIL import Image
from moviepy.editor import VideoClip, AudioFileClip
from moviepy.audio.fx.all import audio_loop
import tempfile
import random
import os
import librosa
import gc
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Recursive-Cut-Photo by Loop507", layout="wide")

if 'v_path' not in st.session_state: st.session_state.v_path = None
if 'r_path' not in st.session_state: st.session_state.r_path = None

def resize_to_format(img, format_type):
    h, w = img.shape[:2]
    if format_type == "16:9 (Orizzontale)": target_w, target_h = 1280, 720
    elif format_type == "9:16 (Verticale)": target_w, target_h = 720, 1280
    else: target_w, target_h = 1080, 1080
    
    aspect_target = target_w / target_h
    aspect_img = w / h
    if aspect_img > aspect_target:
        new_w = int(h * aspect_target)
        start_x = (w - new_w) // 2
        img_cropped = img[:, start_x:start_x+new_w]
    else:
        new_h = int(w / aspect_target)
        start_y = (h - new_h) // 2
        img_cropped = img[start_y:start_y+new_h, :]
    return cv2.resize(img_cropped, (target_w, target_h))

def generate_master(up_m1, up_m2, up_trit, up_aud, orientation, strand_val, max_limit, k_p, m1_s, m1_e, m2_s, m2_e, p_m1_ctrl, p_m2_ctrl, start_c, end_c, format_type, inc_master, rand_lines, photo_speed, chaos_val):
    fps = 24
    total_f = int(max_limit * fps)
    prog_bar = st.progress(0)

    # Assets
    img_m1 = resize_to_format(np.array(Image.open(up_m1).convert("RGB")), format_type) if up_m1 else None
    img_m2 = resize_to_format(np.array(Image.open(up_m2).convert("RGB")), format_type) if up_m2 else None
    t_processed = [resize_to_format(np.array(Image.open(f).convert("RGB")), format_type) for f in up_trit] if up_trit else []
    
    pool_imgs = t_processed.copy()
    if inc_master:
        if img_m1 is not None: pool_imgs.append(img_m1)
        if img_m2 is not None: pool_imgs.append(img_m2)
    if not pool_imgs: pool_imgs = [np.zeros((720, 1280, 3), dtype=np.uint8)]
    
    h, w = pool_imgs[0].shape[:2]
    
    # Audio Analysis
    audio_envelope = np.ones(total_f)
    audio_peak = 0.0
    temp_aud_path = None
    if up_aud:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as t_aud:
            t_aud.write(up_aud.read())
            temp_aud_path = t_aud.name
        y, sr = librosa.load(temp_aud_path, sr=22050, mono=True, duration=max_limit)
        rms = librosa.feature.rms(y=y)[0]
        audio_peak = float(np.max(rms))
        audio_envelope = np.interp(np.linspace(0, len(rms)-1, total_f), np.arange(len(rms)), rms / (rms.max() + 1e-6))

    cached_picks = {}

    def make_frame(t):
        f = int(t * fps)
        if f >= total_f: f = total_f - 1
        prog_bar.progress(f / total_f)
        prog = t / max_limit
        
        # Logica Transizione Statistica
        if prog < start_c:
            prob_m1, prob_m2, prob_c = 1.0, 0.0, 0.0
        elif prog > end_c:
            prob_m1, prob_m2, prob_c = 0.0, 1.0, 0.0
        else:
            t_rel = (prog - start_c) / (end_c - start_c)
            prob_m1 = (1.0 - t_rel) * (p_m1_ctrl / 100.0)
            prob_m2 = t_rel * (p_m2_ctrl / 100.0)
            prob_c = (chaos_val / 100.0)
            total_p = prob_m1 + prob_m2 + prob_c + 1e-6
            prob_m1, prob_m2, prob_c = prob_m1/total_p, prob_m2/total_p, prob_c/total_p

        # Power Curve & Magnetismo
        mid = 0.5
        v_base = (k_p['sv'] + (prog/mid)*(k_p['pv']-k_p['sv'])) if prog <= mid else (k_p['pv'] + ((prog-mid)/mid)*(k_p['ev']-k_p['pv']))
        val = (v_base / 100.0) * audio_envelope[f]
        mag1 = (m1_s + prog * (m1_e - m1_s)) / 100
        mag2 = (m2_s + prog * (m2_e - m2_s)) / 100
        dist_mult = 1.0 - np.clip(mag1 + mag2, 0, 0.95)

        def pick():
            interval = max(1, int(fps / photo_speed))
            key = f // interval
            if key in cached_picks and random.random() > 0.1:
                return cached_picks[key]
            
            r = random.random()
            if img_m1 is not None and r < prob_m1: res = img_m1
            elif img_m2 is not None and r < (prob_m1 + prob_m2): res = img_m2
            else: res = random.choice(pool_imgs)
            
            cached_picks[key] = res
            return res

        if orientation == "Nessun Effetto": return pick()

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        def get_b(max_d):
            res, c = [], 0
            while c < max_d:
                sw = random.randint(max(2, int(strand_val*0.6)), int(strand_val*1.4)) if rand_lines else strand_val
                res.append((c, min(c+sw, max_d)))
                c += sw
            return res

        if orientation == "Orizzontale":
            for s, e in get_b(h):
                target = pick(); shift = int(random.uniform(-500, 500) * val * dist_mult)
                frame[s:e, :] = np.roll(target[s:e, :], shift, axis=1)
        elif orientation == "Verticale":
            for s, e in get_b(w):
                target = pick(); shift = int(random.uniform(-500, 500) * val * dist_mult)
                frame[:, s:e] = np.roll(target[:, s:e], shift, axis=0)
        elif orientation == "Mix (H+V)":
            for s, e in get_b(h):
                target = pick(); shift = int(random.uniform(-500, 500) * val * dist_mult)
                frame[s:e, :] = np.roll(target[s:e, :], shift, axis=1)
            for s, e in get_b(w):
                if random.random() > 0.5:
                    shift_v = int(random.uniform(-400, 400) * val * dist_mult)
                    frame[:, s:e] = np.roll(frame[:, s:e], shift_v, axis=0)
        elif orientation == "Mosaico":
            # RIPRISTINATO ESATTAMENTE COME CODICE 4.0 OK
            for bh in get_b(h):
                for bv in get_b(w):
                    target = pick()
                    shift = int(random.uniform(-400, 400) * val * dist_mult)
                    frame[bh[0]:bh[1], bv[0]:bv[1]] = np.roll(target[bh[0]:bh[1], bv[0]:bv[1]], shift, axis=random.choice([0,1]))
        return frame

    clip = VideoClip(make_frame, duration=max_limit)
    if temp_aud_path:
        audio_clip = AudioFileClip(temp_aud_path)
        if audio_clip.duration < max_limit: audio_clip = audio_loop(audio_clip, duration=max_limit)
        else: audio_clip = audio_clip.set_duration(max_limit)
        clip = clip.set_audio(audio_clip)
    
    v_out = tempfile.mktemp(suffix=".mp4")
    clip.write_videofile(v_out, codec="libx264", audio_codec="aac" if up_aud else None, fps=fps, bitrate="5000k", logger=None)
    
    # --- REPORT BRUTALISTA ---
    report_text = f"""[SLICE_PHOTO_DISSECTION] // VOL_01 // H.264 // DATA_FRAGMENT

:: STILE: Minimalismo Computazionale / Dissezione Brutalista
:: MOTORE: recursive_cut_pro [v7.8]
:: EFFETTO: Recursive Strand Shift (Reattivo)
:: ANALISI: RMS Signal Analysis / Dynamic Slicing
:: PROCESSO: Frammentazione Ricorsiva / Magnetismo Forzato

"L'immagine è stata smontata. Il codice ne ha riscritto la struttura."

---
> TECHNICAL LOG SHEET:
* Asset Pool: {len(pool_imgs)} foto dissezionate
* Rendering: {total_f} frame totali generati
* Geometria: {orientation} @ {strand_val}px
* Power Curve: Start {k_p['sv']}% | Peak {k_p['pv']}% | End {k_p['ev']}%
* Magnetismo: Inizio Snap @ {max_limit * start_c:.1f}s (Pull {m1_s}%)
* Audio Peak: {audio_peak:.4f} normalized

> Regia e Algoritmo: Loop507

#Loop507 #SlicePhoto #StrandShift #DigitalAnatomy #SignalCorruption #BrutalistArt 
#ComputationalMinimalism #DataDestruction #ExperimentalVideo #GlitchArt"""

    r_out = tempfile.mktemp(suffix=".txt")
    with open(r_out, "w") as f: f.write(report_text)
    if temp_aud_path: os.remove(temp_aud_path)
    gc.collect()
    return v_out, r_out

# --- INTERFACCIA ---
st.title("Recursive-Cut-Photo by Loop507 🔪")
c1, c2, c3 = st.columns([1, 1.2, 1])

with c1:
    st.subheader("🖼️ Assets")
    up_m1 = st.file_uploader("MASTER 1 (Start)", type=["jpg","png","jpeg"])
    m1_s = st.slider("M1 Start Magnetism", 0, 100, 100)
    m1_e = st.slider("M1 End Magnetism", 0, 100, 0)
    st.divider()
    up_m2 = st.file_uploader("MASTER 2 (End)", type=["jpg","png","jpeg"])
    m2_s = st.slider("M2 Start Magnetism", 0, 100, 0)
    m2_e = st.slider("M2 End Magnetism", 0, 100, 100)
    st.divider()
    up_t = st.file_uploader("CALDERONE", type=["jpg","png","jpeg"], accept_multiple_files=True)
    up_a = st.file_uploader("AUDIO", type=["mp3","wav"])
    inc_m = st.toggle("Includi Master nel Calderone", value=True)

with c2:
    st.subheader("✂️ Controllo Frame & Transizione")
    p_m1 = st.slider("Percentuale Frame M1 (Presenza)", 0, 100, 100)
    p_m2 = st.slider("Percentuale Frame M2 (Presenza)", 0, 100, 100)
    st.divider()
    start_t = st.slider("Inizio Dissoluzione (%)", 0, 50, 10) / 100.0
    end_t = st.slider("Fine Ricomposizione (%)", 50, 100, 90) / 100.0
    st.divider()
    chaos = st.slider("🌀 Chaos level", 0, 100, 50)
    c_n = chaos / 100.0
    k_params = {'sv': int(85*(1-c_n)+2), 'pv': min(int(100*(1-c_n)+5),100), 'ev': int(75*(1-c_n)+2)}
    speed = st.slider("Photo Speed (fps)", 1, 24, 6)
    lines = st.slider("Strand (px)", 1, 500, 45)
    rand_l = st.toggle("Dynamic Slicing", value=True)
    mode = st.radio("Geometria", ["Orizzontale", "Verticale", "Mix (H+V)", "Mosaico", "Nessun Effetto"])

with c3:
    st.subheader("🎬 Rendering")
    fmt = st.selectbox("Format", ["16:9 (Orizzontale)", "9:16 (Verticale)", "1:1 (Quadrato)"])
    dur = st.number_input("Durata (sec)", 1, 120, 10)
    
    if st.button("🚀 AVVIA DISSEZIONE"):
        v, r = generate_master(up_m1, up_m2, up_t, up_a, mode, lines, dur, k_params, m1_s, m1_e, m2_s, m2_e, p_m1, p_m2, start_t, end_t, fmt, inc_m, rand_l, speed, chaos)
        st.session_state.v_path, st.session_state.r_path = v, r

    if st.session_state.v_path:
        st.video(st.session_state.v_path)
        st.download_button("💾 DOWNLOAD VIDEO", open(st.session_state.v_path, "rb"), "video_dissection.mp4")
        if st.session_state.r_path:
            with open(st.session_state.r_path, "r") as f: r_txt = f.read()
            st.text_area("📄 TECHNICAL REPORT", r_txt, height=480)
            st.download_button("📄 SCARICA REPORT", r_txt, "report_dissection.txt")
