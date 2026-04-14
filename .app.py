import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import random
import librosa
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

st.set_page_config(page_title="Recursive Cut Pro - Loop507", layout="wide")

# --- LOAD IMAGE (NO RESIZE) ---
def load_image(img_file):
    return np.array(Image.open(img_file).convert("RGB"))

# --- STREAM RENDER ---
def render_video_stream(frame_func, w, h, fps, total_f, audio_path=None):

    output_path = tempfile.mktemp(suffix=".mp4")

    writer = FFMPEG_VideoWriter(
        output_path,
        (w, h),
        fps=fps,
        codec="libx264",
        bitrate="4000k",
        audiofile=audio_path
    )

    for f in range(total_f):
        frame = frame_func(f)
        writer.write_frame(frame)

    writer.close()
    return output_path


# --- MAIN GENERATOR ---
def generate_master(
    up_master, up_trit, up_aud,
    orientation, strand_val,
    max_limit, k_p, o_p,
    inc_master, rand_lines,
    photo_speed, chaos_val
):

    fps = 20
    total_f = int(max_limit * fps)

    # --- IMAGES ---
    m_img = load_image(up_master) if up_master else None

    if up_trit:
        t_processed = [load_image(f) for f in up_trit]
    else:
        t_processed = [m_img] if m_img is not None else [np.zeros((720, 1280, 3), dtype=np.uint8)]

    pool_imgs = t_processed.copy()
    if m_img is not None and inc_master:
        pool_imgs.append(m_img)

    h, w = pool_imgs[0].shape[:2]

    # --- SLIDE STRUTTURE PERSISTENTI ---
    def get_bounds(max_dim):
        b = []
        curr = 0
        while curr < max_dim:
            s_w = random.randint(max(2, int(strand_val * 0.1)), int(strand_val * 2)) if rand_lines else strand_val
            if curr + s_w > max_dim:
                s_w = max_dim - curr
            b.append((curr, int(curr + s_w)))
            curr += s_w
        return b

    bounds_h = get_bounds(h)
    bounds_v = get_bounds(w)

    # --- AUDIO ---
    audio_envelope = np.ones(total_f)
    audio_path = None

    if up_aud:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as t:
            up_aud.seek(0)
            t.write(up_aud.read())
            audio_path = t.name

        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=max_limit)
        rms = librosa.feature.rms(y=y)[0]
        rms_norm = rms / (rms.max() + 1e-6)

        audio_envelope = np.interp(
            np.linspace(0, len(rms_norm)-1, total_f),
            np.arange(len(rms_norm)),
            rms_norm
        )

    # --- FRAME GENERATOR ---
    def generate_frame(f):

        curr_s = f / fps
        mid = max_limit / 2

        val = (k_p['sv'] + (f/(total_f/2))*(k_p['pv']-k_p['sv']))/100 if curr_s <= mid else (k_p['pv'] + ((curr_s-mid)/mid)*(k_p['ev']-k_p['pv']))/100
        val *= audio_envelope[f]

        magnet_prob = 0.0
        dist_mult = 1.0

        if m_img is not None and curr_s > o_p['start_fade']:
            t_fade = (curr_s - o_p['start_fade']) / (max_limit - o_p['start_fade'])
            magnet_prob = min(1.0, t_fade * (o_p['final_v'] / 100))
            dist_mult = 1.0 - magnet_prob

        frames_per_photo = max(1, fps // photo_speed)
        active_img = pool_imgs[(f // frames_per_photo) % len(pool_imgs)]

        frame = np.zeros((h, w, 3), dtype=np.uint8)

        def pick():
            return m_img if (m_img is not None and random.random() < magnet_prob) else active_img

        base_shift = int(np.sin(f * 0.1) * 300 * val * dist_mult)

        if orientation == "Orizzontale":
            for i, (start, end) in enumerate(bounds_h):
                target = pick()
                shift = base_shift + int(i * chaos_val * val)
                frame[start:end, :] = np.roll(target[start:end, :], shift, axis=1)

        elif orientation == "Verticale":
            for i, (start, end) in enumerate(bounds_v):
                target = pick()
                shift = base_shift + int(i * chaos_val * val)
                frame[:, start:end] = np.roll(target[:, start:end], shift, axis=0)

        else:
            frame = pick()

        return frame

    # --- RENDER ---
    video_path = render_video_stream(
        generate_frame,
        w, h,
        fps,
        total_f,
        audio_path=audio_path
    )

    return video_path


# ================= UI COMPLETA =================

st.title("Recursive Cut Pro 🚀")

col1, col2 = st.columns(2)

with col1:
    up_m = st.file_uploader("FOTO MASTER", type=["jpg","png"])
    up_t = st.file_uploader("CALDERONE", type=["jpg","png"], accept_multiple_files=True)
    up_a = st.file_uploader("AUDIO", type=["mp3","wav"])

with col2:
    orientation = st.selectbox("Geometria", ["Orizzontale","Verticale","Full"])
    dur = st.slider("Durata", 1, 30, 10)
    lines = st.slider("Spessore linee", 5, 200, 40)
    chaos = st.slider("Chaos", 1, 50, 10)
    photo_speed = st.slider("Velocità cambio immagini", 1, 12, 6)

st.subheader("Keyframe Motion")
col3, col4, col5 = st.columns(3)
sv = col3.slider("Start Value", 0, 100, 10)
pv = col4.slider("Peak Value", 0, 100, 80)
ev = col5.slider("End Value", 0, 100, 20)

st.subheader("Output Magnet")
col6, col7 = st.columns(2)
fade_start = col6.slider("Start Fade", 0, 30, 5)
final_val = col7.slider("Final Strength", 0, 100, 100)

inc_master = st.checkbox("Include Master nel mix", True)
rand_lines = st.checkbox("Linee casuali", False)

if st.button("GENERA"):
    video = generate_master(
        up_m, up_t, up_a,
        orientation,
        lines,
        dur,
        {'sv':sv,'pv':pv,'ev':ev},
        {'start_fade':fade_start,'final_v':final_val},
        inc_master,
        rand_lines,
        photo_speed,
        chaos
    )

    st.video(video)
