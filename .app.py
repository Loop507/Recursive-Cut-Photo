import streamlit as st
import numpy as np
import cv2
from PIL import Image
from moviepy.editor import VideoClip, AudioFileClip
import tempfile
import random
import os
import librosa
import gc
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Recursive-Cut-Photo by Loop507", layout="wide")

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

def generate_master(up_m1, up_m2, up_trit, up_aud, orientation, strand_val, max_limit, k_p, o_p1, o_p2, format_type, inc_master, rand_lines, photo_speed, chaos_val):
    fps = 24
    total_f = int(max_limit * fps)
    prog_bar = st.progress(0)
    status_text = st.empty()

    # --- ASSET SETUP ---
    img_m1 = None
    if up_m1:
        img_m1 = resize_to_format(np.array(Image.open(up_m1).convert("RGB")), format_type)
    
    img_m2 = None
    if up_m2:
        img_m2 = resize_to_format(np.array(Image.open(up_m2).convert("RGB")), format_type)
    
    t_count = len(up_trit) if up_trit else 0
    if up_trit:
        t_processed = [resize_to_format(np.array(Image.open(f).convert("RGB")), format_type) for f in up_trit]
    else:
        # Fallback se non c'è calderone
        t_processed = [img_m1] if img_m1 is not None else [np.zeros((720, 1280, 3), dtype=np.uint8)]

    pool_imgs = t_processed.copy()
    if inc_master:
        if img_m1 is not None: pool_imgs.append(img_m1)
        if img_m2 is not None: pool_imgs.append(img_m2)
    
    h, w = pool_imgs[0].shape[:2]
    
    def get_bounds(max_dim):
        b = []
        curr = 0
        while curr < max_dim:
            s_w = random.randint(max(2, int(strand_val * 0.1)), int(strand_val * 2)) if rand_lines else strand_val
            if curr + s_w > max_dim: s_w = max_dim - curr
            b.append((curr, int(curr + s_w)))
            curr += s_w
        return b

    # --- ANALISI AUDIO ---
    audio_envelope = np.ones(total_f)
    a_info = {"max": 0.0}
    temp_aud_path = None

    if up_aud:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as t_aud:
            up_aud.seek(0)
            t_aud.write(up_aud.read())
            temp_aud_path = t_aud.name
        
        y, sr = librosa.load(temp_aud_path, sr=22050, mono=True, duration=max_limit)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        a_info["max"] = float(rms.max())
        rms_norm = rms / (rms.max() + 1e-6)
        audio_envelope = np.interp(np.linspace(0, len(rms_norm)-1, total_f), np.arange(len(rms_norm)), rms_norm)

    # --- GENERATORE DI FRAME (Motore v7.8 - Dual Master Update) ---
    def make_frame(t):
        f = int(t * fps)
        if f >= total_f: f = total_f - 1
        
        prog_bar.progress(f / total_f)
        status_text.text(f"🚀 Dissecting Signal: {f}/{total_f} frames")

        curr_bounds_h = get_bounds(h)
        curr_bounds_v = get_bounds(w)
        
        mid = max_limit / 2
        if t <= mid:
            val = (k_p['sv'] + (t/mid)*(k_p['pv']-k_p['sv']))/100
        else:
            val = (k_p['pv'] + ((t-mid)/mid)*(k_p['ev']-k_p['pv']))/100
        
        val *= audio_envelope[f]
        
        # LOGICA DUAL MAGNETISM (Transizione X)
        prob_m1 = 0.0
        prob_m2 = 0.0
        
        # Magnetismo Master 1 (Decrescente o controllato)
        if img_m1 is not None:
            # Calcolo basato sul tempo
            prog_m1 = np.clip((t / max_limit), 0, 1)
            # Inversione: parte da Start V e va verso Final V
            prob_m1 = (o_p1['start_v']/100) + prog_m1 * ((o_p1['final_v'] - o_p1['start_v'])/100)

        # Magnetismo Master 2 (Crescente o controllato)
        if img_m2 is not None:
            prog_m2 = np.clip((t / max_limit), 0, 1)
            prob_m2 = (o_p2['start_v']/100) + prog_m2 * ((o_p2['final_v'] - o_p2['start_v'])/100)

        frames_per_photo = max(1, fps // photo_speed)
        active_pool_img = pool_imgs[(f // frames_per_photo) % len(pool_imgs)]
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        def pick():
            r = random.random()
            if img_m1 is not None and r < prob_m1:
                return img_m1
            elif img_m2 is not None and r < (prob_m1 + prob_m2):
                return img_m2
            return active_pool_img

        # Applicazione trasformazione
        dist_mult = 1.0 - np.clip(prob_m1 + prob_m2, 0, 1)

        if orientation == "Orizzontale":
            for start, end in curr_bounds_h:
                target = pick()
                shift = int(random.uniform(-500, 500) * val * dist_mult)
                frame[start:end, :] = np.roll(target[start:end, :], shift, axis=1)
        elif orientation == "Verticale":
            for start, end in curr_bounds_v:
                target = pick()
                shift = int(random.uniform(-500, 500) * val * dist_mult)
                frame[:, start:end] = np.roll(target[:, start:end], shift, axis=0)
        elif orientation == "Mosaico":
            for bh in curr_bounds_h:
                for bv in curr_bounds_v:
                    target = pick()
                    shift = int(random.uniform(-400, 400) * val * dist_mult)
                    if random.random() > 0.5:
                        line_h = np.roll(target[bh[0]:bh[1], :], shift, axis=1)
                        frame[bh[0]:bh[1], bv[0]:bv[1]] = line_h[:, bv[0]:bv[1]]
                    else:
                        line_v = np.roll(target[:, bv[0]:bv[1]], shift, axis=0)
                        frame[bh[0]:bh[1], bv[0]:bv[1]] = line_v[bh[0]:bh[1], :]
        elif orientation == "Mix (H+V)":
            for start, end in curr_bounds_h:
                target = pick()
                shift = int(random.uniform(-500, 500) * val * dist_mult)
                frame[start:end, :] = np.roll(target[start:end, :], shift, axis=random.choice([0,1]))
        else:
            frame = pick()

        return frame

    # --- EXPORT ---
    clip = VideoClip(make_frame, duration=max_limit)
    if temp_aud_path:
        audio_clip = AudioFileClip(temp_aud_path).set_duration(max_limit)
        clip = clip.set_audio(audio_clip)
    
    v_out = tempfile.mktemp(suffix=".mp4")
    clip.write_videofile(v_out, codec="libx264", audio_codec="aac" if up_aud else None, fps=fps, bitrate="5000k", logger=None)
    
    clip.close()
    if up_aud: audio_clip.close()
    if temp_aud_path and os.path.exists(temp_aud_path): os.remove(temp_aud_path)
    gc.collect()

    # --- REPORT ---
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_text = f"""
╔════════════════════════════════════════════════════════════════╗
  Recursive-Cut-Photo by Loop507 - OFFICIAL LOG
  Generated on: {ts}
╚════════════════════════════════════════════════════════════════╝

[SLICE_PHOTO_DISSECTION] // VOL_01 // H.264 // DUAL_MASTER_TRANSITION

:: STILE: Minimalismo Computazionale / Dissezione Brutalista
:: MOTORE: recursive_cut_pro [v7.8]
:: TRANSITION: Dual Anchor Point (M1 -> M2)
:: PROCESSO: Frammentazione Ricorsiva / Cross-Magnetism

---
> TECHNICAL LOG SHEET:
* Rendering: {total_f} frame totali generati
* Geometria: {orientation} @ {strand_val}px
* Magnetismo M1: {o_p1['start_v']}% -> {o_p1['final_v']}%
* Magnetismo M2: {o_p2['start_v']}% -> {o_p2['final_v']}%
* Audio Peak: {a_info['max']:.4f} normalized

> Regia e Algoritmo: Loop507
"""
    r_out = tempfile.mktemp(suffix=".txt")
    with open(r_out, "w", encoding="utf-8") as f_rep:
        f_rep.write(report_text)
    
    return v_out, r_out

# --- INTERFACCIA UTENTE ---
if 'v_p' not in st.session_state: 
    st.session_state.v_p, st.session_state.r_p = None, None

st.title("Recursive-Cut-Photo by Loop507 🔪")
col1, col2, col3 = st.columns([1, 1.2, 1])

with col1:
    st.subheader("🖼️ Master Assets")
    
    st.markdown("**Master 1 (Origine)**")
    up_m1 = st.file_uploader("FOTO MASTER 1", type=["jpg","png","jpeg"], key="m1")
    m1_start = st.slider("M1 Magnetismo Inizio %", 0, 100, 100)
    m1_end = st.slider("M1 Magnetismo Fine %", 0, 100, 0)
    
    st.divider()
    
    st.markdown("**Master 2 (Destinazione)**")
    up_m2 = st.file_uploader("FOTO MASTER 2", type=["jpg","png","jpeg"], key="m2")
    m2_start = st.slider("M2 Magnetismo Inizio %", 0, 100, 0)
    m2_end = st.slider("M2 Magnetismo Fine %", 0, 100, 100)
    
    st.divider()
    up_t = st.file_uploader("CALDERONE (Pool frammenti)", type=["jpg","png","jpeg"], accept_multiple_files=True)
    up_a = st.file_uploader("AUDIO (Analisi RMS)", type=["mp3","wav"])
    inc_m = st.toggle("Includi Master nel Calderone", value=True)

with col2:
    st.subheader("✂️ Algoritmo di Taglio")
    chaos = st.slider("🌀 Chaos vs Order", 0, 100, 50)
    c_n = chaos / 100.0
    k_params = {'sv': int(85*(1-c_n)+2), 'pv': min(int(100*(1-c_n)+5),100), 'ev': int(75*(1-c_n)+2)}
    
    with st.expander("⚙️ Fine-Tune Potenza"):
        sv = st.slider("Start Power", 0, 100, k_params['sv'])
        pv = st.slider("Peak Power", 0, 100, k_params['pv'])
        ev = st.slider("End Power", 0, 100, k_params['ev'])
        k_params = {'sv': sv, 'pv': pv, 'ev': ev}

    st.divider()
    photo_speed = st.slider("🎞️ Frame Rate Foto (fps)", 1, 24, 6)
    lines = st.slider("Spessore Strand (px)", 1, 500, 45)
    rand_l = st.toggle("Dynamic Slicing (Tagli Random)", value=False)
    dir_type = st.radio("Geometria di Dissezione", ["Orizzontale", "Verticale", "Mosaico", "Mix (H+V)"])

with col3:
    st.subheader("🎬 Rendering")
    fmt = st.selectbox("Aspect Ratio", ["16:9 (Orizzontale)", "9:16 (Verticale)", "1:1 (Quadrato)"])
    dur = st.number_input("Durata Totale (sec)", 1, 120, 10)
    
    if st.button("🚀 AVVIA DISSEZIONE"):
        if up_m1 or up_m2 or up_t:
            v, r = generate_master(up_m1, up_m2, up_t, up_a, dir_type, lines, dur, k_params, 
                                   {'start_v':m1_start,'final_v':m1_end}, 
                                   {'start_v':m2_start,'final_v':m2_end}, 
                                   fmt, inc_m, rand_l, photo_speed, chaos)
            st.session_state.v_p, st.session_state.r_p = v, r
        else:
            st.error("Errore: Carica almeno un asset.")

    if st.session_state.v_p:
        st.video(st.session_state.v_p)
        st.download_button("💾 SCARICA VIDEO", open(st.session_state.v_p, "rb"), "recursive_cut.mp4")
        st.download_button("📄 SCARICA REPORT TECNICO", open(st.session_state.r_p, "rb"), "recursive_report.txt")
