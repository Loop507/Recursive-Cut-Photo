import streamlit as st
import numpy as np
import cv2
from PIL import Image
from moviepy.editor import ImageSequenceClip, AudioFileClip
import tempfile
import random
import os
import librosa

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

def generate_master(up_master, up_trit, up_aud, mode, orientation, strand_val, max_limit, k_p, o_p, format_type, inc_master, rand_lines, photo_speed=6):
    fps = 24
    total_f = int(max_limit * fps)
    
    prog_bar = st.progress(0)
    status_text = st.empty()

    m_img = None
    if up_master:
        m_img = resize_to_format(np.array(Image.open(up_master).convert("RGB")), format_type)
    
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

    curr_bounds_h = get_bounds(h)
    curr_bounds_v = get_bounds(w)

    # --- PRE-PROCESSING AUDIO REACTIVE ---
    audio_envelope = np.ones(total_f)
    if up_aud:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as t_pre:
            up_aud.seek(0)
            t_pre.write(up_aud.read())
            pre_aud_path = t_pre.name
        y, sr = librosa.load(pre_aud_path, sr=None, mono=True, duration=max_limit)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_norm = rms / (rms.max() + 1e-6)
        frame_indices = np.linspace(0, len(rms_norm) - 1, total_f)
        audio_envelope = np.interp(frame_indices, np.arange(len(rms_norm)), rms_norm)
        up_aud.seek(0)

    final_frames = []

    for f in range(total_f):
        prog_bar.progress(f / total_f)
        status_text.text(f"🚀 Rendering Frame: {f}/{total_f}")
        
        if rand_lines and f % 2 == 0:
            curr_bounds_h = get_bounds(h)
            curr_bounds_v = get_bounds(w)
        
        curr_s = f / fps
        mid = max_limit / 2
        val = (k_p['sv'] + (f/(total_f/2))*(k_p['pv']-k_p['sv']))/100 if curr_s <= mid else (k_p['pv'] + ((curr_s-mid)/mid)*(k_p['ev']-k_p['pv']))/100
        val *= audio_envelope[f]
        
        magnet_prob = 0.0
        dist_mult = 1.0 
        if m_img is not None and curr_s > o_p['start_fade']:
            t_fade = (curr_s - o_p['start_fade']) / (max_limit - o_p['start_fade'])
            magnet_prob = min(1.0, t_fade * (o_p['final_v'] / 100))
            if (max_limit - curr_s) < 0.3: magnet_prob = 1.0; dist_mult = 0.0
            else: dist_mult = 1.0 - magnet_prob

        frames_per_photo = max(1, fps // photo_speed)
        active_pool_img = pool_imgs[(f // frames_per_photo) % len(pool_imgs)]
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        def pick(prob):
            return m_img if (m_img is not None and random.random() < prob) else active_pool_img

        # --- LOGICA GEOMETRIE CORRETTA --- 
        
        # 1. NESSUNO (STUTTER)
        if orientation == "Nessuno (Foto Intere)":
            target = pick(magnet_prob)
            shift = int(random.uniform(-400, 400) * val * dist_mult)
            frame = np.roll(target, shift, axis=1)
        
        # 2. MOSAICO (Griglia statica di foto diverse)
        elif orientation == "Mosaico":
            for bh in curr_bounds_h:
                for bv in curr_bounds_v:
                    target = pick(magnet_prob)
                    frame[bh[0]:bh[1], bv[0]:bv[1]] = target[bh[0]:bh[1], bv[0]:bv[1]]
        
        # 3. MIX (H+V) - Tagli incrociati NON sovrapposti per ogni cella
        elif orientation == "Mix (H+V)":
            for bh in curr_bounds_h:
                for bv in curr_bounds_v:
                    target = pick(magnet_prob)
                    shift = int(random.uniform(-350, 350) * val * dist_mult)
                    if random.random() > 0.5:
                        line_h = np.roll(target[bh[0]:bh[1], :], shift, axis=1)
                        frame[bh[0]:bh[1], bv[0]:bv[1]] = line_h[:, bv[0]:bv[1]]
                    else:
                        line_v = np.roll(target[:, bv[0]:bv[1]], shift, axis=0)
                        frame[bh[0]:bh[1], bv[0]:bv[1]] = line_v[bh[0]:bh[1], :]

        # 4. ORIZZONTALE / VERTICALE (Ripristino logica strisce intere Codice 6) [cite: 4]
        else:
            if orientation == "Orizzontale":
                for start, end in curr_bounds_h:
                    target = pick(magnet_prob)
                    shift = int(random.uniform(-350, 350) * val * dist_mult)
                    frame[start:end, :] = np.roll(target[start:end, :], shift, axis=1)
            elif orientation == "Verticale":
                for start, end in curr_bounds_v:
                    target = pick(magnet_prob)
                    shift = int(random.uniform(-350, 350) * val * dist_mult)
                    frame[:, start:end] = np.roll(target[:, start:end], shift, axis=0)

        final_frames.append(frame)

    prog_bar.empty()
    status_text.text("💾 Generazione file finali...")
    clip = ImageSequenceClip(final_frames, fps=fps)
    if up_aud:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as t_aud:
            up_aud.seek(0)
            t_aud.write(up_aud.read()); aud_path = t_aud.name
        clip = clip.set_audio(AudioFileClip(aud_path).subclip(0, min(AudioFileClip(aud_path).duration, max_limit)))
    
    v_out = tempfile.mktemp(suffix=".mp4")
    clip.write_videofile(v_out, codec="libx264", audio_codec="aac" if up_aud else None, fps=fps, bitrate="6000k", logger=None)
    
    report_text = f"--- LOOP507 REPORT 7.1 ---\nStile: {orientation}\nGriglie Random: {rand_lines}\nFrame: {total_f}"
    l_out = tempfile.mktemp(suffix=".txt")
    with open(l_out, "w") as f: f.write(report_text)
    
    return v_out, l_out

# --- INTERFACCIA ---
if 'v_p' not in st.session_state: st.session_state.v_p = None
if 'r_p' not in st.session_state: st.session_state.r_p = None

st.title("Recursive Cut Pro - Loop507 7.1 🚀")
col1, col2, col3 = st.columns([1, 1.2, 1])

with col1:
    st.subheader("🖼️ Assets")
    up_m = st.file_uploader("FOTO MASTER", type=["jpg","png","jpeg"])
    up_t = st.file_uploader("CALDERONE", type=["jpg","png","jpeg"], accept_multiple_files=True)
    up_a = st.file_uploader("AUDIO", type=["mp3","wav"])
    inc_m = st.toggle("Master nel Calderone", value=True)
    m_f = st.slider("Magnetismo Finale %", 0, 100, 100)
    m_s = st.slider("Inizio Snap (sec)", 0.0, 10.0, 7.0)

with col2:
    st.subheader("✂️ Parametri")
    chaos_order = st.slider("🌀 Caos → Ordine", 0, 100, 50)
    chaos_norm = chaos_order / 100.0
    sv = int(80 * (1 - chaos_norm) + 2)
    pv = min(int(100 * (1 - chaos_norm) + 5), 100)
    ev = int(70 * (1 - chaos_norm) + 2)

    with st.expander("⚙️ Override manuale sv/pv/ev"):
        sv = st.slider("Start Power", 0, 100, sv)
        pv = st.slider("Peak Power", 0, 100, pv)
        ev = st.slider("End Power", 0, 100, ev)

    st.divider()
    photo_speed = st.slider("🎞️ Velocità cambio foto (fps)", 1, 24, 6)
    st.divider()
    lines = st.slider("Spessore Base (px)", 1, 500, 45)
    rand_l = st.toggle("Dimensioni Random (DINAMICHE)", value=False)
    dir_type = st.radio("Geometria", ["Orizzontale", "Verticale", "Mosaico", "Mix (H+V)", "Nessuno (Foto Intere)"])

with col3:
    st.subheader("🎬 Export")
    fmt = st.selectbox("Formato", ["16:9 (Orizzontale)", "9:16 (Verticale)", "1:1 (Quadrato)"])
    dur = st.number_input("Durata (sec)", 1, 60, 10)
    
    if st.button("🚀 GENERA"):
        if up_m or up_t:
            v, r = generate_master(up_m, up_t, up_a, "Recursive", dir_type, lines, dur, {'sv':sv,'pv':pv,'ev':ev}, {'start_fade':m_s,'final_v':m_f}, fmt, inc_m, rand_l, photo_speed)
            st.session_state.v_p, st.session_state.r_p = v, r
        else: st.error("Carica foto!")

    if st.session_state.v_p:
        st.video(st.session_state.v_p)
        c1, c2 = st.columns(2)
        with c1: st.download_button("💾 VIDEO", open(st.session_state.v_p, "rb"), "video.mp4")
        with c2: st.download_button("📄 LOG", open(st.session_state.r_p, "rb"), "report.txt")
