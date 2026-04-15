Hai ragione, nell'ultima versione ho involontariamente rimosso pezzi fondamentali che avevamo costruito nelle ore precedenti (magnetismo start/end, la logica del report completo e l'integrazione audio-reattiva totale).

Ho ricostruito il codice **completo al 100%**, inserendo i nuovi slider per le **Percentuali di Frame** (Decadenza M1 e Nascita M2) e mantenendo tutto il resto (Power Curve, Magnetismo, Report Brutalista e Analisi Audio).

### Cosa è incluso in questa versione:
1.  **Magnetismo Start/End:** Per forzare l'attrazione verso le master.
2.  **Percentuale Frame M1/M2:** I nuovi comandi per decidere quanti "pezzi" di immagine sopravvivono.
3.  **Comandi Frame (10%-90%):** Per gestire quando inizia e finisce il collasso.
4.  **Power Curve Dinamica:** Legata al Chaos level e all'Audio.
5.  **Report Integrale:** Con tutti i dati tecnici richiesti.

```python
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

def generate_master(up_m1, up_m2, up_trit, up_aud, orientation, strand_val, max_limit, k_p, m1_mag, m2_mag, p_m1_perc, p_m2_perc, start_c, end_c, format_type, inc_master, rand_lines, chaos_val):
    fps = 24
    total_f = int(max_limit * fps)
    prog_bar = st.progress(0)

    # Caricamento Asset
    img_m1 = resize_to_format(np.array(Image.open(up_m1).convert("RGB")), format_type) if up_m1 else None
    img_m2 = resize_to_format(np.array(Image.open(up_m2).convert("RGB")), format_type) if up_m2 else None
    t_processed = [resize_to_format(np.array(Image.open(f).convert("RGB")), format_type) for f in up_trit] if up_trit else []
    
    pool_imgs = t_processed.copy()
    if inc_master:
        if img_m1 is not None: pool_imgs.append(img_m1)
        if img_m2 is not None: pool_imgs.append(img_m2)
    if not pool_imgs: pool_imgs = [np.zeros((h, w, 3), dtype=np.uint8)]
    
    h, w = (720, 1280) if format_type == "16:9 (Orizzontale)" else (1280, 720) if format_type == "9:16 (Verticale)" else (1080, 1080)
    
    # Analisi Audio
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

    def make_frame(t):
        f = int(t * fps)
        if f >= total_f: f = total_f - 1
        prog_bar.progress(f / total_f)
        rel_t = t / max_limit
        
        # --- LOGICA PERCENTUALE FRAME & TRANSIZIONE ---
        if rel_t < start_c:
            raw_m1, raw_m2, raw_c = 1.0, 0.0, 0.0
        elif rel_t > end_c:
            raw_m1, raw_m2, raw_c = 0.0, 1.0, 0.0
        else:
            # Transizione normalizzata tra gli slider temporali
            tr = (rel_t - start_c) / (end_c - start_c)
            raw_m1 = (1.0 - tr) * (p_m1_perc / 100.0)
            raw_m2 = tr * (p_m2_perc / 100.0)
            raw_c = (chaos_val / 100.0)

        # Calcolo magnetismo istantaneo
        m1_m = (m1_mag['s'] + rel_t * (m1_mag['e'] - m1_mag['s'])) / 100.0
        m2_m = (m2_mag['s'] + rel_t * (m2_mag['e'] - m2_mag['s'])) / 100.0
        
        # Pesi finali della "Zuppa"
        w_m1 = raw_m1 * (1.0 + m1_m)
        w_m2 = raw_m2 * (1.0 + m2_m)
        total_w = w_m1 + w_m2 + raw_c + 1e-6
        
        # Power Curve
        mid = 0.5
        v_base = (k_p['sv'] + (rel_t/mid)*(k_p['pv']-k_p['sv'])) if rel_t <= mid else (k_p['pv'] + ((rel_t-mid)/mid)*(k_p['ev']-k_p['pv']))
        val = (v_base / 100.0) * audio_envelope[f]
        
        # Smorzamento agli estremi
        if rel_t < (start_c * 0.5) or rel_t > (end_c + (1-end_c)*0.5): val *= 0.05

        def pick():
            r = random.random()
            if img_m1 is not None and r < (w_m1 / total_w): return img_m1
            if img_m2 is not None and r < ((w_m1 + w_m2) / total_w): return img_m2
            return random.choice(pool_imgs)

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        def get_b(max_d):
            res, c = [], 0
            while c < max_d:
                sw = random.randint(int(strand_val*0.7), int(strand_val*1.3)) if rand_lines else strand_val
                res.append((c, min(c+sw, max_d)))
                c += sw
            return res

        if orientation == "Orizzontale":
            for s, e in get_b(h):
                targ = pick(); shift = int(random.uniform(-500, 500) * val)
                frame[s:e, :] = np.roll(targ[s:e, :], shift, axis=1)
        elif orientation == "Verticale":
            for s, e in get_b(w):
                targ = pick(); shift = int(random.uniform(-500, 500) * val)
                frame[:, s:e] = np.roll(targ[:, s:e], shift, axis=0)
        else: # Mix (H+V)
            for s, e in get_b(h):
                targ = pick(); shift = int(random.uniform(-500, 500) * val)
                frame[s:e, :] = np.roll(targ[s:e, :], shift, axis=1)
            for s, e in get_b(w):
                if random.random() > 0.5:
                    frame[:, s:e] = np.roll(frame[:, s:e], int(random.uniform(-300, 300)*val), axis=0)
        return frame

    clip = VideoClip(make_frame, duration=max_limit)
    if up_aud:
        audio_clip = AudioFileClip(temp_aud_path)
        clip = clip.set_audio(audio_clip.set_duration(max_limit))
    
    v_out = tempfile.mktemp(suffix=".mp4")
    clip.write_videofile(v_out, codec="libx264", audio_codec="aac" if up_aud else None, fps=fps, bitrate="5000k", logger=None)
    
    # REPORT COMPLETO RICHIESTO
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
* Caos Range: {int(start_c*100)}% -> {int(end_c*100)}%
* Frame Density: M1 {p_m1_perc}% | M2 {p_m2_perc}%
* Audio Peak: {audio_peak:.4f} normalized
* Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

> Regia e Algoritmo: Loop507

#Loop507 #SlicePhoto #StrandShift #DigitalAnatomy #SignalCorruption #BrutalistArt 
#ComputationalMinimalism #DataDestruction #ExperimentalVideo #GlitchArt"""

    r_out = tempfile.mktemp(suffix=".txt")
    with open(r_out, "w") as f: f.write(report_text)
    if temp_aud_path: os.remove(temp_aud_path)
    gc.collect()
    return v_out, r_out

# --- UI ---
st.title("Recursive-Cut-Photo v7.8 🔪")
c1, c2, c3 = st.columns([1, 1.2, 1])

with c1:
    st.subheader("🖼️ Assets")
    up_m1 = st.file_uploader("MASTER 1 (Start)", type=["jpg","png","jpeg"])
    m1_s = st.slider("M1 Magnetism Start", 0, 100, 100)
    m1_e = st.slider("M1 Magnetism End", 0, 100, 0)
    st.divider()
    up_m2 = st.file_uploader("MASTER 2 (End)", type=["jpg","png","jpeg"])
    m2_s = st.slider("M2 Magnetism Start", 0, 100, 0)
    m2_e = st.slider("M2 Magnetism End", 0, 100, 100)
    st.divider()
    up_t = st.file_uploader("CALDERONE", type=["jpg","png","jpeg"], accept_multiple_files=True)
    up_a = st.file_uploader("AUDIO", type=["mp3","wav"])
    inc_m = st.toggle("Includi Master nel Calderone", value=True)

with c2:
    st.subheader("✂️ Algoritmo & Transizione")
    p_m1 = st.slider("Percentuale Frame M1 (Sopravvivenza)", 0, 100, 100)
    p_m2 = st.slider("Percentuale Frame M2 (Nascita)", 0, 100, 100)
    st.divider()
    start_c = st.slider("Inizio Decadenza (%)", 0, 50, 10) / 100.0
    end_c = st.slider("Fine Ricomposizione (%)", 50, 100, 90) / 100.0
    st.divider()
    chaos_p = st.slider("🌀 Intensità Caos (Power)", 0, 100, 50)
    c_n = chaos_p / 100.0
    kp = {'sv': int(85*(1-c_n)+2), 'pv': min(int(100*(1-c_n)+5), 100), 'ev': int(75*(1-c_n)+2)}
    strand = st.slider("Strand (px)", 1, 500, 45)
    rand_l = st.toggle("Dynamic Slicing", value=True)
    mode = st.radio("Geometria", ["Orizzontale", "Verticale", "Mix (H+V)"])

with c3:
    st.subheader("🎬 Rendering")
    fmt = st.selectbox("Format", ["16:9 (Orizzontale)", "9:16 (Verticale)", "1:1 (Quadrato)"])
    dur = st.number_input("Durata (sec)", 1, 120, 10)
    
    if st.button("🚀 AVVIA DISSEZIONE"):
        v, r = generate_master(up_m1, up_m2, up_t, up_a, mode, strand, dur, kp, {'s':m1_s,'e':m1_e}, {'s':m2_s,'e':m2_e}, p_m1, p_m2, start_c, end_c, fmt, inc_m, rand_l, chaos_p)
        st.session_state.v_path, st.session_state.r_path = v, r

    if st.session_state.v_path:
        st.video(st.session_state.v_path)
        st.download_button("💾 DOWNLOAD VIDEO", open(st.session_state.v_path, "rb"), "video_dissection.mp4")
        if st.session_state.r_path:
            with open(st.session_state.r_path, "r") as f: r_txt = f.read()
            st.text_area("📄 REPORT", r_txt, height=450)
            st.download_button("📄 SCARICA REPORT", r_txt, "report_dissection.txt")
```
