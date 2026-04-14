import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import random
import librosa
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from moviepy.editor import AudioFileClip

st.set_page_config(page_title="Recursive Cut Pro - Loop507", layout="wide")

# --- RESIZE ---
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
        img = img[:, start_x:start_x+new_w]
    else:
        new_h = int(w / aspect_target)
        start_y = (h - new_h) // 2
        img = img[start_y:start_y+new_h, :]

    return cv2.resize(img, (target_w, target_h))


# --- RENDER STREAM ---
def render_video_stream(frame_func, w, h, fps, total_f, audio_path=None, max_limit=10):

    temp_video = tempfile.mktemp(suffix=".mp4")

    writer = FFMPEG_VideoWriter(
        temp_video,
        (w, h),
        fps=fps,
        codec="libx264",
        bitrate="8000k"
    )

    for f in range(total_f):
        frame = frame_func(f)
        writer.write_frame(frame)

    writer.close()

    if audio_path:
        video = AudioFileClip(temp_video)
        audio = AudioFileClip(audio_path)

        final = video.set_audio(
            audio.subclip(0, min(audio.duration, max_limit))
        )

        final_output = tempfile.mktemp(suffix=".mp4")
        final.write_videofile(
            final_output,
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            logger=None
        )

        return final_output

    return temp_video


# --- GENERATE ---
def generate_master(up_master, up_trit, up_aud, orientation, strand_val,
                    max_limit, k_p, o_p, format_type, inc_master,
                    rand_lines, photo_speed, chaos_val):

    fps = 24
    total_f = int(max_limit * fps)

    # --- IMMAGINI ---
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

    # --- BOUNDS ---
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

    # --- AUDIO ANALYSIS ---
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
            dist_mult = 1.0 - magnet_prob

        frames_per_photo = max(1, fps // photo_speed)
        active_img = pool_imgs[(f // frames_per_photo) % len(pool_imgs)]

        frame = np.zeros((h, w, 3), dtype=np.uint8)

        def pick():
            return m_img if (m_img is not None and random.random() < magnet_prob) else active_img

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

        else:
            frame = pick()

        return frame

    # --- RENDER ---
    video_path = render_video_stream(
        generate_frame,
        w, h,
        fps,
        total_f,
        audio_path=audio_path,
        max_limit=max_limit
    )

    return video_path


# --- UI ---
st.title("Recursive Cut Pro 🚀")

up_m = st.file_uploader("FOTO MASTER", type=["jpg","png"])
up_t = st.file_uploader("CALDERONE", type=["jpg","png"], accept_multiple_files=True)
up_a = st.file_uploader("AUDIO", type=["mp3","wav"])

dur = st.slider("Durata", 1, 60, 10)
orientation = st.selectbox("Geometria", ["Orizzontale","Verticale"])
lines = st.slider("Spessore", 5, 200, 40)

if st.button("GENERA"):
    video = generate_master(
        up_m, up_t, up_a,
        orientation,
        lines,
        dur,
        {'sv':10,'pv':80,'ev':20},
        {'start_fade':5,'final_v':100},
        "16:9 (Orizzontale)",
        True,
        False,
        6,
        50
    )

    st.video(video)
