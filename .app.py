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
st.set_page_config(page_title="Recursive Cut Pro - Loop507", layout="wide")

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

def generate_master(up_master, up_trit, up_aud, orientation, strand_val, max_limit, k_p, o_p, format_type, inc_master, rand_lines, photo_speed, chaos_val):
    fps = 24
    total_f = int(max_limit * fps)
    prog_bar = st.progress(0)
    status_text = st.empty()

    # --- ASSET SETUP ---
    m_img = None
    m_name = "Assente"
    if up_master:
        m_img = resize_to_format(np.array(Image.open(up_master).convert("RGB")), format_type)
        m_name = "Presente (Master)"
    
    t_count = len(up_trit) if up_trit else 0
    if up_trit:
        t_processed = [resize_to_format(np.array(Image.open(f).convert("RGB")), format_type) for f in up_trit]
    else:
        t_processed = [m_img] if m_img is not None else [np.zeros((720, 1280, 3), dtype=np.uint8)]

    pool_imgs = t_processed.copy()
    if m_img is not None and inc_master:
        pool_imgs.append(m_img)
    
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
    a_info = {"min": 0.0, "max": 0.0, "mean": 0.0, "active": "No"}
    temp_aud_path = None

    if up_aud:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as t_aud:
            up_aud.seek(0)
            t_aud.write(up_aud.read())
            temp_aud_path = t_aud.name
        
        y, sr = librosa.load(temp_aud_path, sr=22050, mono=True, duration=max_limit)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        a_info = {"min": float(rms.min()), "max": float(rms.max()), "mean": float(rms.mean()), "active": "Attivo (RMS Analysis)"}
        rms_norm = rms / (rms.max() + 1e-6)
        audio_envelope = np.interp(np.linspace(0, len(rms_norm)-1, total_f), np.arange(len(rms_norm)), rms_norm)

    # --- GENERATORE DI FRAME ---
    def make_frame(t):
        f = int(t * fps)
        if f >= total_f: f = total_f - 1
        
        prog_bar.progress(f / total_f)
        status_text.text(f"🚀 Rendering Frame: {f}/{total_f}")

        curr_bounds_h = get_bounds(h)
        curr_bounds_v = get_bounds(w)
        
        curr_s = t
        mid = max_limit / 2
        
        val = (k_p['sv'] + (f/(total_f/2))*(k_p['pv']-k_p['sv']))/100 if curr_s <= mid else (k_p['pv'] + ((curr_s-mid)/mid)*(k_p['ev']-k_p['pv']))/100
        val *= audio_envelope[f]
        
        magnet_prob = 0.0
        dist_mult = 1.0 
        if m_img is not None and curr_s > o_p['start_fade']:
            t_fade = (curr_s - o_p['start_fade']) / (max_limit - o_p['start_fade'])
            magnet_prob = min(1.0, t_fade * (o_p['final_v'] / 100))
            if (max_limit - curr_s) < 0.25:
                magnet_prob = 1.0; dist_mult = 0.0
            else:
                dist_mult = 1.0 - magnet_prob

        frames_per_photo = max(1, fps // photo_speed)
        active_pool_img = pool_imgs[(f // frames_per_photo) % len(pool_imgs)]
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        def pick():
            return m_img if (m_img is not None and random.random() < magnet_prob) else active_pool_img

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
                direction = random.choice([-1, 1])
                shift = int(random.uniform(50, 500) * val * dist_mult) * direction
                if random.random() > 0.5:
                    frame[start:end, :] = np.roll(target[start:end, :], shift, axis=1)
                else:
                    frame[start:end, :] = np.roll(target[start:end, :], shift, axis=0)
        else:
            frame = pick()

        return frame

    # --- EXPORT VIDEO ---
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

    # --- REPORT PROFESSIONALE ---
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = f"""
╔════════════════════════════════════════════════════════════════╗
  RECURSIVE CUT PRO v7.8 - TECHNICAL DATA SHEET
  Generated on: {ts}
╚════════════════════════════════════════════════════════════════╝

[1] SPECIFICHE VIDEO
Pixels          : {w}x{h} ({format_type})
Duration        : {max_limit} seconds
Frame Rate      : {fps} FPS
Total Frames    : {total_f}
Bitrate Target  : 5000 kbps

[2] ANALISI ASSET
Master Image    : {m_name}
Pool Size       : {t_count} images
Master in Pool  : {"Attivo" if inc_master else "Disattivato"}
Audio Track     : {a_info['active']}
Audio RMS Peak  : {a_info['max']:.4f}

[3] CONFIGURAZIONE EFFETTI
Geometry        : {orientation}
Base Thickness  : {strand_val}px
Dynamic Slicing : {"Sì" if rand_lines else "No"}
Photo Cycling   : {photo_speed} fps
Chaos/Order     : {chaos_val}%

[4] LOGICA DI MAGNETISMO
Start Fade      : {o_p['start_fade']} sec
Final Pull      : {o_p['final_v']}%
End Snap        : 0.25 sec before end

[5] POWER CURVE (Potenza Glitch)
Start Power     : {k_p['sv']}%
Peak Power      : {k_p['pv']}%
End Power       : {k_p['ev']}%

------------------------------------------------------------------
Final Output    : {os.path.basename(v_out)}
Encoder         : H.264 / AAC (Optimized for Streamlit)
------------------------------------------------------------------
          Generated by Recursive Cut Pro - Loop507
    """
    r_out = tempfile.mktemp(suffix=".txt")
    with open(r_out, "w", encoding="utf-8") as f_rep:
        f_rep.write(report_content)
    
    return v_out, r_out

# --- INTERFACCIA ---
if 'v_p' not in st.session_state: st.session_state.v_p, st.session_state.r_p = None, None

st.title("Recursive Cut Pro 7.8 🚀")
col1, col2, col3 = st.columns([1, 1.2, 1])

with col1:
    st.subheader("🖼️ Assets & Magnetismo")
    up_m = st.file_uploader("FOTO MASTER", type=["jpg","png","jpeg"])
    up_t = st.file_uploader("CALDERONE", type=["jpg","png","jpeg"], accept_multiple_files=True)
    up_a = st.file_uploader("AUDIO", type=["mp3","wav"])
    st.divider()
    m_f = st.slider("Magnetismo Master Finale %", 0, 100, 100)
    m_s = st.slider("Inizio Ritorno Master (sec)", 0.0, 10.0, 7.0)
    inc_m = st.toggle("Usa Master nel Calderone", value=True)

with col2:
    st.subheader("✂️ Parametri Glitch")
    chaos = st.slider("🌀 Bilanciamento Caos → Ordine", 0, 100, 50)
    c_n = chaos / 100.0
    k_params = {'sv': int(85*(1-c_n)+2), 'pv': min(int(100*(1-c_n)+5),100), 'ev': int(75*(1-c_n)+2)}
    
    with st.expander("⚙️ Override Potenza"):
        sv = st.slider("Start", 0, 100, k_params['sv'])
        pv = st.slider("Peak", 0, 100, k_params['pv'])
        ev = st.slider("End", 0, 100, k_params['ev'])
        k_params = {'sv': sv, 'pv': pv, 'ev': ev}

    st.divider()
    photo_speed = st.slider("🎞️ Velocità Cambio Foto (fps)", 1, 24, 6)
    lines = st.slider("Spessore Tagli (px)", 1, 500, 45)
    rand_l = st.toggle("Tagli Dinamici", value=False)
    dir_type = st.radio("Geometria", ["Orizzontale", "Verticale", "Mosaico", "Mix (H+V)", "Nessuno (Foto Intere)"])

with col3:
    st.subheader("🎬 Export")
    fmt = st.selectbox("Formato", ["16:9 (Orizzontale)", "9:16 (Verticale)", "1:1 (Quadrato)"])
    dur = st.number_input("Durata (sec)", 1, 120, 10)
    
    if st.button("🚀 GENERA VIDEO"):
        if up_m or up_t:
            v, r = generate_master(up_m, up_t, up_a, dir_type, lines, dur, k_params, {'start_fade':m_s,'final_v':m_f}, fmt, inc_m, rand_l, photo_speed, chaos)
            st.session_state.v_p, st.session_state.r_p = v, r
        else: st.error("Carica foto!")

    if st.session_state.v_p:
        st.video(st.session_state.v_p)
        st.download_button("💾 VIDEO", open(st.session_state.v_p, "rb"), "loop_78.mp4")
        st.download_button("📄 REPORT TECNICO", open(st.session_state.r_p, "rb"), "report_pro_78.txt")
