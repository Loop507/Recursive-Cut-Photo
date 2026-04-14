import streamlit as st
import numpy as np
import cv2
from PIL import Image
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
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


# 🔥 NUOVO RENDER (STREAMING - NO CRASH)
def render_stream(frame_func, w, h, fps, total_f, audio_path=None):

    v_out = tempfile.mktemp(suffix=".mp4")

    writer = FFMPEG_VideoWriter(
        v_out,
        (w, h),
        fps=fps,
        codec="libx264",
        bitrate="6000k",
        audiofile=audio_path
    )

    for f in range(total_f):
        frame = frame_func(f)
        writer.write_frame(frame)

    writer.close()
    return v_out


def generate_master(up_master, up_trit, up_aud, orientation, strand_val, max_limit, k_p, o_p, format_type, inc_master, rand_lines, photo_speed, chaos_val):

    fps = 24
    total_f = int(max_limit * fps)

    prog_bar = st.progress(0)
    status_text = st.empty()

    # --- ASSET ---
    m_img = None
    m_name = "No"
    if up_master:
        m_img = resize_to_format(np.array(Image.open(up_master).convert("RGB")), format_type)
        m_name = "Si"

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
            if curr + s_w > max_dim:
                s_w = max_dim - curr
            b.append((curr, int(curr + s_w)))
            curr += s_w
        return b

    # --- AUDIO ---
    audio_envelope = np.ones(total_f)
    audio_path = None
    a_info = {"min": 0.0, "max": 0.0, "mean": 0.0, "active": "No"}

    if up_aud:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as t:
            up_aud.seek(0)
            t.write(up_aud.read())
            audio_path = t.name

        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=max_limit)
        rms = librosa.feature.rms(y=y)[0]
        a_info = {"min": float(rms.min()), "max": float(rms.max()), "mean": float(rms.mean()), "active": "Si"}

        rms_norm = rms / (rms.max() + 1e-6)
        audio_envelope = np.interp(
            np.linspace(0, len(rms_norm)-1, total_f),
            np.arange(len(rms_norm)),
            rms_norm
        )

    # 🔥 FRAME GENERATOR (IDENTICO)
    def generate_frame(f):

        prog_bar.progress(f / total_f)
        status_text.text(f"🚀 Rendering: {f}/{total_f}")

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
            if (max_limit - curr_s) < 0.25:
                magnet_prob = 1.0
                dist_mult = 0.0
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

    # 🔥 RENDER STREAM
    v_out = render_stream(generate_frame, w, h, fps, total_f, audio_path)

    prog_bar.empty()

    # --- REPORT IDENTICO ---
    report_text = f"""--- LOOP507 REPORT ---

Durata: {max_limit} sec | Frame: {total_f} | FPS: {fps}
Foto Master: {m_name}
Foto nel Calderone: {t_count}
Audio: {a_info['active']}
"""

    r_out = tempfile.mktemp(suffix=".txt")
    with open(r_out, "w") as f_rep:
        f_rep.write(report_text)

    return v_out, r_out
