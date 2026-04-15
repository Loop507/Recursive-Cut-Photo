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
    img_m1 = resize_to_format(np.array(Image.open(up_m1).convert("RGB")), format_type) if up_m1 else None
    img_m2 = resize_to_format(np.array(Image.open(up_m2).convert("RGB")), format_type) if up_m2 else None
    
    t_processed = [resize_to_format(np.array(Image.open(f).convert("RGB")), format_type) for f in up_trit] if up_trit else []
    
    pool_imgs = t_processed.copy()
    if inc_master:
        if img_m1 is not None: pool_imgs.append(img_m1)
        if img_m2 is not None: pool_imgs.append(img_m2)
    
    if not pool_imgs:
        pool_imgs = [np.zeros((720, 1280, 3), dtype=np.uint8)]
    
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

    # --- MOTORE DI DISSEZIONE ---
    def make_frame(t):
        f = int(t * fps)
        if f >= total_f: f = total_f - 1
        prog_bar.progress(f / total_f)
        status_text.text(f"🚀 Dissecting Signal: {f}/{total_f} frames")

        # Curva di potenza glitch originale
        prog = t / max_limit
        mid = 0.5
        val = (k_p['sv'] + (prog/mid)*(k_p['pv']-k_p['sv']))/100 if prog <= mid else (k_p['pv'] + ((prog-mid)/mid)*(k_p['ev']-k_p['pv']))/100
        val *= audio_envelope[f]
        
        # Calcolo magnetismo X
        mag1 = (o_p1['start_v'] + prog * (o_p1['final_v'] - o_p1['start_v'])) / 100
        mag2 = (o_p2['start_v'] + prog * (o_p2['final_v'] - o_p2['start_v'])) / 100
        
        # Mix del calderone (garantisce visibilità dei frammenti)
        frames_per_photo = max(1, fps // photo_speed)
        active_pool_img = pool_imgs[(f // frames_per_photo) % len(pool_imgs)]
        
        def pick():
            r = random.random()
            # Se chaos_val è alto, aumenta la probabilità di pescare dal calderone
            chaos_trigger = (chaos_val / 100.0) * 0.5 
            if r < chaos_trigger: return pool_imgs[random.randint(0, len(pool_imgs)-1)]
            if img_m1 is not None and r < (chaos_trigger + mag1): return img_m1
            if img_m2 is not None and r < (chaos_trigger + mag1 + mag2): return img_m2
            return active_pool_img

        if orientation == "Nessun Effetto":
            return pick()

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        dist_mult = 1.0 - np.clip(mag1 + mag2, 0, 0.95)

        if orientation in ["Orizzontale", "Mix (H+V)"]:
            for s, e in get_bounds(h):
                target = pick()
                shift = int(random.uniform(-500, 500) * val * dist_mult)
                frame[s:e, :] = np.roll(target[s:e, :], shift, axis=1)
        elif orientation == "Verticale":
            for s, e in get_bounds(w):
                target = pick()
                shift = int(random.uniform(-500, 500) * val * dist_mult)
                frame[:, s:e] = np.roll(target[:, s:e], shift, axis=0)
        elif orientation == "Mosaico":
            for bh in get_bounds(h):
                for bv in get_bounds(w):
                    target = pick()
                    shift = int(random.uniform(-400, 400) * val * dist_mult)
                    if random.random() > 0.5:
                        line = np.roll(target[bh[0]:bh[1], :], shift, axis=1)
                        frame[bh[0]:bh[1], bv[0]:bv[1]] = line[:, bv[0]:bv[1]]
                    else:
                        line = np.roll(target[:, bv[0]:bv[1]], shift, axis=0)
                        frame[bh[0]:bh[1], bv[0]:bv[1]] = line[bh[0]:bh[1], :]
        return frame

    # --- EXPORT ---
    clip = VideoClip(make_frame, duration=max_limit)
    if temp_aud_path:
        audio_clip = AudioFileClip(temp_aud_path).set_duration(max_limit)
        clip = clip.set_audio(audio_clip)
    
    v_out = tempfile.mktemp(suffix=".mp4")
    clip.write_videofile(v_out, codec="libx264", audio_codec="aac" if up_aud else None, fps=fps, bitrate="5000k", logger=None)
    
    # --- REPORT ---
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_text = f"Recursive-Cut-Photo by Loop507\nGenerated: {ts}\nMode: {orientation}\nM1 Mag: {o_p1}\nM2 Mag: {o_p2}\nChaos: {chaos_val}"
    r_out = tempfile.mktemp(suffix=".txt")
    with open(r_out, "w") as f_rep: f_rep.write(report_text)
    
    if temp_aud_path and os.path.exists(temp_aud_path): os.remove(temp_aud_path)
    gc.collect()
    return v_out, r_out

# --- INTERFACCIA ---
if 'v_p' not in st.session_state: st.session_state.v_p, st.session_state.r_p = None, None

st.title("Recursive-Cut-Photo by Loop507 🔪")
c1, c2, c3 = st.columns([1, 1.2, 1])

with c1:
    st.subheader("🖼️ Assets")
    up_m1 = st.file_uploader("FOTO MASTER 1 (Origine)", type=["jpg","png","jpeg"])
    m1_vals = st.slider("M1 Magnetismo (Inizio/Fine)", 0, 100, (100, 0))
    st.divider()
    up_m2 = st.file_uploader("FOTO MASTER 2 (Destinazione)", type=["jpg","png","jpeg"])
    m2_vals = st.slider("M2 Magnetismo (Inizio/Fine)", 0, 100, (0, 100))
    st.divider()
    up_t = st.file_uploader("CALDERONE (Frammenti)", type=["jpg","png","jpeg"], accept_multiple_files=True)
    up_a = st.file_uploader("AUDIO (Analisi RMS)", type=["mp3","wav"])
    inc_m = st.toggle("Includi Master nel Calderone", value=True)

with c2:
    st.subheader("✂️ Algoritmo")
    chaos = st.slider("🌀 Chaos vs Order", 0, 100, 50)
    c_n = chaos / 100.0
    k_params = {'sv': int(85*(1-c_n)+2), 'pv': min(int(100*(1-c_n)+5),100), 'ev': int(75*(1-c_n)+2)}
    
    with st.expander("⚙️ Fine-Tune Potenza"):
        sv = st.slider("Start Power", 0, 100, k_params['sv'])
        pv = st.slider("Peak Power", 0, 100, k_params['pv'])
        ev = st.slider("End Power", 0, 100, k_params['ev'])
        k_params = {'sv': sv, 'pv': pv, 'ev': ev}

    st.divider()
    speed = st.slider("🎞️ Photo Speed (fps)", 1, 24, 6)
    lines = st.slider("Strand (px)", 1, 500, 45)
    rand_l = st.toggle("Dynamic Slicing", value=True)
    mode = st.radio("Geometria", ["Orizzontale", "Verticale", "Mosaico", "Mix (H+V)", "Nessun Effetto"])

with c3:
    st.subheader("🎬 Export")
    fmt = st.selectbox("Aspect Ratio", ["16:9 (Orizzontale)", "9:16 (Verticale)", "1:1 (Quadrato)"])
    dur = st.number_input("Durata (sec)", 1, 120, 10)
    
    if st.button("🚀 AVVIA DISSEZIONE"):
        v, r = generate_master(up_m1, up_m2, up_t, up_a, mode, lines, dur, k_params, 
                               {'start_v':m1_vals[0],'final_v':m1_vals[1]}, 
                               {'start_v':m2_vals[0],'final_v':m2_vals[1]}, fmt, inc_m, rand_l, speed, chaos)
        st.session_state.v_p, st.session_state.r_p = v, r

    if st.session_state.v_p:
        st.video(st.session_state.v_p)
        st.download_button("💾 VIDEO", open(st.session_state.v_p, "rb"), "recursive_cut.mp4")
        st.download_button("📄 REPORT", open(st.session_state.r_p, "rb"), "report.txt")
