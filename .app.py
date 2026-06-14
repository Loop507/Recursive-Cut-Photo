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

if 'v_path'  not in st.session_state: st.session_state.v_path  = None
if 'r_path'  not in st.session_state: st.session_state.r_path  = None
if 'file_ts' not in st.session_state: st.session_state.file_ts = None

# --- PRESET GENERE ---
GENRE_PRESETS = {
    "Techno / House":  {"beat_strength": 70, "beat_decay": 20, "onset":  0, "cache": 20, "rhythm": False},
    "Orchestrale":     {"beat_strength": 20, "beat_decay": 80, "onset": 40, "cache": 25, "rhythm": True},
    "Pop / Soul":      {"beat_strength": 50, "beat_decay": 50, "onset": 60, "cache": 35, "rhythm": False},
    "Glitch / Noise":  {"beat_strength": 90, "beat_decay": 10, "onset": 80, "cache": 70, "rhythm": True},
    "Drone / Pad":     {"beat_strength":  0, "beat_decay": 60, "onset": 20, "cache": 15, "rhythm": True},
    "Hip Hop / Jazz":  {"beat_strength": 60, "beat_decay": 35, "onset": 50, "cache": 30, "rhythm": False},
}

def resize_to_format(img, format_type, half_res=False):
    if format_type == "16:9 (Orizzontale)": target_w, target_h = 1280, 720
    elif format_type == "9:16 (Verticale)":  target_w, target_h = 720, 1280
    else:                                     target_w, target_h = 1080, 1080
    if half_res:
        target_w, target_h = target_w // 2, target_h // 2
    h, w = img.shape[:2]
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


def apply_glitch_stripes(src, dst, h, w, orientation, strand_val, rand_lines, val):
    """Applica le strisce glitch da src verso dst con intensità val."""
    strand = max(1, strand_val // 2)

    def get_b(max_d):
        res, c = [], 0
        while c < max_d:
            sw = random.randint(max(2, int(strand*0.6)), int(strand*1.4)) if rand_lines else strand
            res.append((c, min(c+sw, max_d)))
            c += sw
        return res

    frame = np.zeros((h, w, 3), dtype=np.uint8)

    if orientation == "Orizzontale":
        for s, e in get_b(h):
            shift = int(random.uniform(-250, 250) * val)
            frame[s:e, :] = np.roll(src[s:e, :], shift, axis=1)
    elif orientation == "Verticale":
        for s, e in get_b(w):
            shift = int(random.uniform(-250, 250) * val)
            frame[:, s:e] = np.roll(src[:, s:e], shift, axis=0)
    elif orientation == "Mix (H+V)":
        for s, e in get_b(h):
            shift = int(random.uniform(-250, 250) * val)
            frame[s:e, :] = np.roll(src[s:e, :], shift, axis=1)
        for s, e in get_b(w):
            if random.random() > 0.5:
                frame[:, s:e] = np.roll(frame[:, s:e], int(random.uniform(-200, 200) * val), axis=0)
    elif orientation == "Mosaico":
        shift_h = int(random.uniform(-250, 250) * val)
        shift_v = int(random.uniform(-250, 250) * val)
        rolled = np.roll(np.roll(src, shift_h, axis=1), shift_v, axis=0)
        for bh in get_b(h):
            for bw in get_b(w):
                if random.random() > 0.5:
                    frame[bh[0]:bh[1], bw[0]:bw[1]] = rolled[bh[0]:bh[1], bw[0]:bw[1]]
                else:
                    frame[bh[0]:bh[1], bw[0]:bw[1]] = src[bh[0]:bh[1], bw[0]:bw[1]]
    else:
        frame = src.copy()

    alpha = np.clip(val, 0.0, 1.0)
    return (frame * alpha + dst * (1.0 - alpha)).astype(np.uint8)


def apply_stripe_window(bg_frame, calder_clean, calder_glitch, h, w,
                        stripes, stripe_orientation, stripe_glitch,
                        stripe_reverse=False):
    """
    Compone il frame finale in modalità striscia.
    stripe_reverse=False: strisce = Calderone, resto = Master (sfondo fermo)
    stripe_reverse=True:  strisce = Master (fermo), resto = Calderone
    stripe_orientation può essere "Orizzontale", "Verticale" o "Mix H+V"
    """
    src_calder = calder_glitch if stripe_glitch else calder_clean

    if stripe_reverse:
        # base = Calderone, le strisce mostrano il Master fermo
        out = src_calder.copy()
        src_stripe = bg_frame
    else:
        # base = Master fermo, le strisce mostrano il Calderone
        out = bg_frame.copy()
        src_stripe = src_calder

    def _apply_h(stripes_list):
        for (pos, size) in stripes_list:
            y0 = int(np.clip(pos / 100.0, 0.0, 1.0) * h)
            y1 = int(np.clip((pos + size) / 100.0, 0.0, 1.0) * h)
            if y1 > y0:
                out[y0:y1, :] = src_stripe[y0:y1, :]

    def _apply_v(stripes_list):
        for (pos, size) in stripes_list:
            x0 = int(np.clip(pos / 100.0, 0.0, 1.0) * w)
            x1 = int(np.clip((pos + size) / 100.0, 0.0, 1.0) * w)
            if x1 > x0:
                out[:, x0:x1] = src_stripe[:, x0:x1]

    if stripe_orientation == "Orizzontale":
        _apply_h(stripes)
    elif stripe_orientation == "Verticale":
        _apply_v(stripes)
    elif stripe_orientation == "Mix H+V":
        # strisce pari = orizzontali, dispari = verticali
        h_stripes = [s for i, s in enumerate(stripes) if i % 2 == 0]
        v_stripes = [s for i, s in enumerate(stripes) if i % 2 == 1]
        _apply_h(h_stripes)
        _apply_v(v_stripes)

    return out


def generate_master(up_m1, up_m2, up_trit, up_aud,
                    orientation, strand_val, max_limit,
                    chaos_val, photo_speed, format_type,
                    m1_end, m2_start,
                    rand_lines,
                    beat_sync, genre,
                    seq_mode,
                    slideshow_mode, slide_hold, slide_trans, slide_trans_type,
                    stripe_mode=False, stripes=None, stripe_orientation="Orizzontale",
                    stripe_bg="Master 1", stripe_glitch=False, stripe_reverse=False):

    fps = 24
    total_f = int(max_limit * fps)
    prog_bar = st.progress(0)

    def load_img_full(f):
        f.seek(0)
        return resize_to_format(np.array(Image.open(f).convert("RGB")), format_type, half_res=False)
    def load_img_half(f):
        f.seek(0)
        return resize_to_format(np.array(Image.open(f).convert("RGB")), format_type, half_res=True)

    img_m1 = load_img_full(up_m1) if up_m1 else None
    img_m2 = load_img_full(up_m2) if up_m2 else None
    pool_imgs = [load_img_half(f) for f in up_trit] if up_trit else []
    if not pool_imgs:
        pool_imgs = [np.zeros((360, 640, 3), dtype=np.uint8)]

    h, w = pool_imgs[0].shape[:2]

    img_m1_half = load_img_half(up_m1) if up_m1 else None
    img_m2_half = load_img_half(up_m2) if up_m2 else None

    if format_type == "16:9 (Orizzontale)": out_w, out_h = 1280, 720
    elif format_type == "9:16 (Verticale)":  out_w, out_h = 720, 1280
    else:                                     out_w, out_h = 1080, 1080

    # --- sfondo fisso per stripe mode: stesso shape ESATTO del pool (h, w) ---
    if stripe_mode and stripes:
        if stripe_bg == "Master 1" and up_m1 is not None:
            up_m1.seek(0)
            _bg_raw = resize_to_format(np.array(Image.open(up_m1).convert("RGB")), format_type, half_res=True)
            stripe_bg_img = cv2.resize(_bg_raw, (w, h))
        elif stripe_bg == "Master 2" and up_m2 is not None:
            up_m2.seek(0)
            _bg_raw = resize_to_format(np.array(Image.open(up_m2).convert("RGB")), format_type, half_res=True)
            stripe_bg_img = cv2.resize(_bg_raw, (w, h))
        else:
            stripe_bg_img = pool_imgs[0].copy()
    else:
        stripe_bg_img = None

    # --- AUDIO ANALYSIS ---
    audio_envelope  = np.ones(total_f)
    beat_envelope   = np.zeros(total_f)
    onset_envelope  = np.zeros(total_f)
    rhythm_envelope = None
    audio_peak      = 0.0
    beat_count      = 0
    temp_aud_path   = None
    bs, bd, op, bc  = 0, 50, 0, 30
    rhythm_tracking = False

    if up_aud:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as t:
            t.write(up_aud.read())
            temp_aud_path = t.name

        y, sr = librosa.load(temp_aud_path, sr=22050, mono=True, duration=max_limit)

        rms = librosa.feature.rms(y=y)[0]
        audio_peak = float(np.max(rms))
        audio_envelope = np.interp(
            np.linspace(0, len(rms)-1, total_f),
            np.arange(len(rms)), rms / (rms.max() + 1e-6)
        )

        if beat_sync and not slideshow_mode:
            p = GENRE_PRESETS[genre]
            bs = p["beat_strength"]
            bd = p["beat_decay"]
            op = p["onset"]
            bc = p["cache"]
            rhythm_tracking = p["rhythm"]

            if bs > 0:
                _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                beat_times = librosa.frames_to_time(beat_frames, sr=sr)
                beat_count = len(beat_times)
                decay_rate = 1.0 - (bd / 100.0) * 0.98
                for bt in beat_times:
                    bf = int(bt * fps)
                    for df in range(min(int(fps * 0.5), total_f - bf)):
                        beat_envelope[bf + df] = max(beat_envelope[bf + df], decay_rate ** df)

            if op > 0:
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
                for ot in librosa.frames_to_time(onset_frames, sr=sr):
                    of = int(ot * fps)
                    if of < total_f:
                        onset_envelope[of] = 1.0

        if rhythm_tracking:
            tempogram = librosa.feature.tempogram(y=y, sr=sr)
            tempo_local = tempogram.max(axis=0).astype(float)
            tempo_local /= (tempo_local.max() + 1e-6)
            stft = np.abs(librosa.stft(y))
            flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
            flux /= (flux.max() + 1e-6)
            min_len = min(len(tempo_local), len(flux))
            combined = (tempo_local[:min_len] * 0.5 + flux[:min_len] * 0.5)
            combined /= (combined.max() + 1e-6)
            rhythm_envelope = np.interp(
                np.linspace(0, len(combined)-1, total_f),
                np.arange(len(combined)), combined
            )
            kernel = np.ones(fps // 2) / (fps // 2)
            rhythm_envelope = np.convolve(rhythm_envelope, kernel, mode='same')
            rhythm_envelope = np.clip(rhythm_envelope / (rhythm_envelope.max() + 1e-6), 0.0, 1.0)

    # --- CACHE ---
    MAX_CACHE = 400
    cached_picks = {}
    cache_keys_order = []

    def cache_set(key, val):
        if key not in cached_picks:
            if len(cache_keys_order) >= MAX_CACHE:
                old = cache_keys_order.pop(0)
                cached_picks.pop(old, None)
            cache_keys_order.append(key)
        cached_picks[key] = val

    # --- POWER CURVE ---
    c_n = chaos_val / 100.0
    sv = int(85 * (1 - c_n) + 2)
    pv = min(int(100 * (1 - c_n) + 5), 100)
    ev = int(75 * (1 - c_n) + 2)

    # =========================================================
    # MODALITÀ SLIDESHOW
    # =========================================================
    if slideshow_mode and pool_imgs:
        slide_cycle = slide_hold + slide_trans
        n_photos = len(pool_imgs)

        def make_frame_slideshow(t):
            f = int(t * fps)
            if f >= total_f: f = total_f - 1
            prog_bar.progress(f / total_f)

            cycle_pos = t % slide_cycle
            if seq_mode:
                idx_cur = int(t / slide_cycle) % n_photos
            else:
                cycle_idx = int(t / slide_cycle)
                random.seed(cycle_idx * 9999)
                idx_cur = random.randint(0, n_photos - 1)
                random.seed()

            idx_next = (idx_cur + 1) % n_photos
            img_cur  = pool_imgs[idx_cur]
            img_next = pool_imgs[idx_next]

            if cycle_pos < slide_hold:
                if stripe_mode and stripes and stripe_bg_img is not None:
                    out_frame = apply_stripe_window(stripe_bg_img, img_cur, img_cur, h, w,
                                                    stripes, stripe_orientation, False, stripe_reverse)
                else:
                    out_frame = img_cur
                return cv2.resize(out_frame, (out_w, out_h))
            else:
                trans_prog = (cycle_pos - slide_hold) / max(slide_trans, 0.001)
                trans_prog = np.clip(trans_prog, 0.0, 1.0)

                if slide_trans_type == "Glitch Burst":
                    if trans_prog < 0.5:
                        intensity = trans_prog * 2.0
                        base = img_cur
                        dest = img_next
                    else:
                        intensity = (1.0 - trans_prog) * 2.0
                        base = img_next
                        dest = img_next
                    glitched = apply_glitch_stripes(base, dest, h, w, orientation, strand_val, rand_lines, intensity)
                    if stripe_mode and stripes and stripe_bg_img is not None:
                        out_frame = apply_stripe_window(stripe_bg_img, base, glitched, h, w,
                                                        stripes, stripe_orientation, stripe_glitch, stripe_reverse)
                    else:
                        out_frame = glitched
                else:  # Dissolve Glitchato
                    intensity = np.sin(trans_prog * np.pi)
                    blend = (img_cur * (1.0 - trans_prog) + img_next * trans_prog).astype(np.uint8)
                    glitched = apply_glitch_stripes(blend, blend, h, w, orientation, strand_val, rand_lines, intensity)
                    if stripe_mode and stripes and stripe_bg_img is not None:
                        out_frame = apply_stripe_window(stripe_bg_img, blend, glitched, h, w,
                                                        stripes, stripe_orientation, stripe_glitch, stripe_reverse)
                    else:
                        out_frame = glitched

                return cv2.resize(out_frame, (out_w, out_h))

        clip = VideoClip(make_frame_slideshow, duration=max_limit)

    # =========================================================
    # MODALITÀ NORMALE
    # =========================================================
    else:
        def make_frame(t):
            f = int(t * fps)
            if f >= total_f: f = total_f - 1
            prog_bar.progress(f / total_f)
            prog = t / max_limit

            has_masters = (img_m1 is not None) and (img_m2 is not None)

            if has_masters:
                if prog <= m1_end:
                    _ramp_m1 = np.clip(prog / m1_end if m1_end > 0.001 else 1.0, 0.0, 1.0)
                    if _ramp_m1 < 0.02:
                        return cv2.resize(img_m1, (out_w, out_h))
                    _m1_prob = 1.0 - _ramp_m1
                    def pick():
                        key = f // max(1, int(fps / photo_speed))
                        if key in cached_picks and random.random() > 0.1:
                            return cached_picks[key]
                        if random.random() < _m1_prob:
                            res = img_m1_half
                        else:
                            idx = (key % len(pool_imgs)) if seq_mode else None
                            res = pool_imgs[idx] if seq_mode else random.choice(pool_imgs)
                        cache_set(key, res)
                        return res
                elif prog >= m2_start:
                    _span_m2 = 1.0 - m2_start if m2_start < 0.999 else 1e-6
                    _ramp_m2 = np.clip((prog - m2_start) / _span_m2, 0.0, 1.0)
                    if _ramp_m2 > 0.98:
                        return cv2.resize(img_m2, (out_w, out_h))
                    _m2_prob = _ramp_m2
                    def pick():
                        key = f // max(1, int(fps / photo_speed))
                        if key in cached_picks and random.random() > 0.1:
                            return cached_picks[key]
                        if random.random() < _m2_prob:
                            res = img_m2_half
                        else:
                            idx = (key % len(pool_imgs)) if seq_mode else None
                            res = pool_imgs[idx] if seq_mode else random.choice(pool_imgs)
                        cache_set(key, res)
                        return res
                else:
                    def pick():
                        interval = max(1, int(fps / photo_speed))
                        key = f // interval
                        force = onset_envelope[f] > 0 and random.random() < (bc / 100.0) if beat_sync else False
                        if key in cached_picks and not force and random.random() > 0.1:
                            return cached_picks[key]
                        idx = (key % len(pool_imgs)) if seq_mode else None
                        res = pool_imgs[idx] if seq_mode else random.choice(pool_imgs)
                        cache_set(key, res)
                        return res
            else:
                def pick():
                    if rhythm_envelope is not None:
                        speed_mod = max(1, photo_speed * (0.2 + rhythm_envelope[f] * 0.8))
                    else:
                        speed_mod = photo_speed
                    interval = max(1, int(fps / speed_mod))
                    key = f // interval
                    force = onset_envelope[f] > 0 and beat_sync and random.random() < (bc / 100.0) if beat_sync else False
                    if key in cached_picks and not force and random.random() > 0.1:
                        return cached_picks[key]
                    idx = (key % len(pool_imgs)) if seq_mode else None
                    res = pool_imgs[idx] if seq_mode else random.choice(pool_imgs)
                    cache_set(key, res)
                    return res

            has_masters_val = (img_m1 is not None) and (img_m2 is not None)

            if rhythm_envelope is not None:
                val_base = rhythm_envelope[f]
            else:
                mid = 0.5
                v_base = (sv + (prog/mid)*(pv-sv)) if prog <= mid else (pv + ((prog-mid)/mid)*(ev-pv))
                val_base = (v_base / 100.0) * audio_envelope[f]

            if beat_sync and beat_envelope[f] > 0:
                val_base = val_base * (1.0 + beat_envelope[f] * (bs / 100.0))

            if has_masters_val:
                if prog <= m1_end:
                    ramp = prog / m1_end if m1_end > 0.001 else 1.0
                    val = val_base * np.clip(ramp, 0.0, 1.0)
                elif prog >= m2_start:
                    span = 1.0 - m2_start if m2_start < 0.999 else 1e-6
                    ramp = 1.0 - ((prog - m2_start) / span)
                    val = val_base * np.clip(ramp, 0.0, 1.0)
                else:
                    val = val_base
            else:
                val = val_base

            strand = max(1, strand_val // 2)

            def get_b(max_d):
                res, c = [], 0
                while c < max_d:
                    sw = random.randint(max(2, int(strand*0.6)), int(strand*1.4)) if rand_lines else strand
                    res.append((c, min(c+sw, max_d)))
                    c += sw
                return res

            if orientation == "Nessun Effetto":
                if stripe_mode and stripes and stripe_bg_img is not None:
                    calder_clean = pick()
                    return cv2.resize(
                        apply_stripe_window(stripe_bg_img, calder_clean, calder_clean, h, w,
                                            stripes, stripe_orientation, False, stripe_reverse),
                        (out_w, out_h))
                return cv2.resize(pick(), (out_w, out_h))

            # --- Calderone corrente (foto pulita, per la striscia originale) ---
            calder_clean = pick()

            frame = np.zeros((h, w, 3), dtype=np.uint8)

            if orientation == "Orizzontale":
                for s, e in get_b(h):
                    target = pick()
                    shift = int(random.uniform(-250, 250) * val)
                    frame[s:e, :] = np.roll(target[s:e, :], shift, axis=1)
            elif orientation == "Verticale":
                for s, e in get_b(w):
                    target = pick()
                    shift = int(random.uniform(-250, 250) * val)
                    frame[:, s:e] = np.roll(target[:, s:e], shift, axis=0)
            elif orientation == "Mix (H+V)":
                for s, e in get_b(h):
                    target = pick()
                    shift = int(random.uniform(-250, 250) * val)
                    frame[s:e, :] = np.roll(target[s:e, :], shift, axis=1)
                for s, e in get_b(w):
                    if random.random() > 0.5:
                        shift_v = int(random.uniform(-200, 200) * val)
                        frame[:, s:e] = np.roll(frame[:, s:e], shift_v, axis=0)
            elif orientation == "Mosaico":
                target = pick()
                shift_h = int(random.uniform(-250, 250) * val)
                shift_v = int(random.uniform(-250, 250) * val)
                rolled = np.roll(np.roll(target, shift_h, axis=1), shift_v, axis=0)
                bh_list = get_b(h)
                bw_list = get_b(w)
                for bh in bh_list:
                    for bw in bw_list:
                        if random.random() > 0.5:
                            frame[bh[0]:bh[1], bw[0]:bw[1]] = rolled[bh[0]:bh[1], bw[0]:bw[1]]
                        else:
                            frame[bh[0]:bh[1], bw[0]:bw[1]] = target[bh[0]:bh[1], bw[0]:bw[1]]

            # --- Stripe mode: componi sfondo + striscia ---
            if stripe_mode and stripes and stripe_bg_img is not None:
                frame = apply_stripe_window(stripe_bg_img, calder_clean, frame, h, w,
                                            stripes, stripe_orientation, stripe_glitch, stripe_reverse)

            return cv2.resize(frame, (out_w, out_h))

        clip = VideoClip(make_frame, duration=max_limit)

    # --- AUDIO ---
    if temp_aud_path:
        audio_clip = AudioFileClip(temp_aud_path)
        if audio_clip.duration < max_limit: audio_clip = audio_loop(audio_clip, duration=max_limit)
        else: audio_clip = audio_clip.set_duration(max_limit)
        clip = clip.set_audio(audio_clip)

    # --- NOMI FILE con timestamp condiviso ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"video_dissection_{ts}"

    v_out = tempfile.mktemp(suffix=".mp4")
    clip.write_videofile(v_out, codec="libx264", audio_codec="aac" if up_aud else None,
                         fps=fps, bitrate="5000k", logger=None)

    rhythm_on = beat_sync and not slideshow_mode and GENRE_PRESETS[genre]["rhythm"]

    slide_info = ""
    if slideshow_mode:
        slide_info = f"""
* MODALITÀ: SLIDESHOW LENTO
* Durata foto: {slide_hold}s | Transizione: {slide_trans}s
* Tipo transizione: {slide_trans_type}
* Sequenza: {'ORDINATA' if seq_mode else 'RANDOM'}"""

    stripe_info = ""
    if stripe_mode and stripes:
        stripe_info = f"""
* STRISCE SELETTIVE: {len(stripes)} striscia/e ({stripe_orientation})
* Sfondo: {stripe_bg} | Striscia: {'GLITCHATA' if stripe_glitch else 'ORIGINALE'}"""

    report_text = f"""[SLICE_PHOTO_DISSECTION] // VOL_01 // H.264 // DATA_FRAGMENT
:: MOTORE: recursive_cut_pro [v9.3]
:: EFFETTO: Recursive Strand Shift
:: ANALISI: RMS / Beat Sync / Rhythm Tracking
:: PROCESSO: Frammentazione Ricorsiva
"L'immagine e' stata smontata. Il codice ne ha riscritto la struttura."

> TECHNICAL LOG SHEET:
* File: {base_name}
* Asset Pool: {len(pool_imgs)} foto
* Rendering: {total_f} frame @ {fps}fps
* Geometria: {orientation} @ {strand_val}px
* Chaos: {chaos_val}% | Photo Speed: {photo_speed}fps
* M1 sparisce a: {int(m1_end*100)}% | M2 appare a: {int(m2_start*100)}%
* Audio Peak: {audio_peak:.4f}
* Beat Sync: {'ON' if beat_sync and not slideshow_mode else 'OFF (Slideshow)' if slideshow_mode else 'OFF'}
* Power Curve: {'BYPASSED' if rhythm_on else 'ON'}
* Sequenza Calderone: {'ORDINATA' if seq_mode else 'RANDOM'}{slide_info}{stripe_info}

> Regia e Algoritmo: Loop507

#glitchart #slicephoto #strandshift #digitalanatomy #signalcorruption #brutalistart
#computationalminimalism #datadestruction #experimentalvideo"""

    r_out = tempfile.mktemp(suffix=".txt")
    with open(r_out, "w") as f: f.write(report_text)
    if temp_aud_path: os.remove(temp_aud_path)
    gc.collect()
    return v_out, r_out, base_name


# =====================================================================
# INTERFACCIA
# =====================================================================
st.title("Recursive-Cut-Photo by Loop507 🔪")
c1, c2, c3 = st.columns([1, 1.2, 1])

with c1:
    st.subheader("🖼️ Assets")
    up_m1 = st.file_uploader("MASTER 1 — inizio", type=["jpg","png","jpeg"])
    up_m2 = st.file_uploader("MASTER 2 — fine",   type=["jpg","png","jpeg"])
    st.divider()
    up_t = st.file_uploader("CALDERONE", type=["jpg","png","jpeg"], accept_multiple_files=True)
    st.divider()
    up_a = st.file_uploader("AUDIO", type=["mp3","wav"])

with c2:
    st.subheader("✂️ Controllo")

    has_masters = (up_m1 is not None) and (up_m2 is not None)
    if has_masters:
        st.caption("Transizione M1 → Calderone → M2")
        m1_end_t  = st.slider("M1 sparisce a (%)",  0, 100, 50) / 100.0
        m2_start_t = st.slider("M2 appare a (%)", 0, 100, 60) / 100.0
        if m2_start_t < m1_end_t:
            st.caption("⚠️ M2 appare prima che M1 finisca — si sovrappongono.")
        st.divider()
    else:
        m1_end_t, m2_start_t = 0.5, 0.6
        if up_m1 or up_m2:
            st.caption("⚠️ Carica entrambi i Master per attivare la transizione.")

    chaos  = st.slider("🌀 Chaos", 0, 100, 50)
    speed  = st.slider("⚡ Photo Speed (fps)", 1, 24, 6)
    lines  = st.slider("📐 Strand (px)", 1, 500, 45)
    rand_l = st.toggle("Dynamic Slicing", value=True)
    mode   = st.radio("Geometria", ["Orizzontale", "Verticale", "Mix (H+V)", "Mosaico", "Nessun Effetto"])

    st.divider()

    # ---- STRISCE SELETTIVE ----
    stripe_mode = st.toggle("🎯 Strisce Selettive", value=False,
        help="Sfondo fisso (Master) + finestre che mostrano il Calderone in movimento.")

    stripes            = []
    stripe_orientation = "Orizzontale"
    stripe_bg          = "Master 1"
    stripe_glitch      = False
    stripe_reverse     = False

    if stripe_mode:
        stripe_orientation = st.radio("Orientamento strisce",
            ["Orizzontale", "Verticale", "Mix H+V"], horizontal=True,
            help="Mix H+V: strisce pari = orizzontali, dispari = verticali")

        # scelta sfondo
        bg_opts = []
        if up_m1: bg_opts.append("Master 1")
        if up_m2: bg_opts.append("Master 2")
        if not bg_opts: bg_opts = ["Master 1"]
        stripe_bg = st.radio("🖼️ Sfondo fisso", bg_opts, horizontal=True,
            help="La foto ferma su cui si aprono le strisce")

        col_tog1, col_tog2 = st.columns(2)
        with col_tog1:
            stripe_glitch = st.toggle("⚡ Striscia glitchata", value=False,
                help="OFF = striscia mostra foto Calderone pulita. ON = glitchata.")
        with col_tog2:
            stripe_reverse = st.toggle("🔄 Reverse", value=False,
                help="Inverte: strisce ferme (Master), tutto il resto Calderone in movimento.")

        st.divider()

        n_stripes = st.number_input("Numero di strisce", min_value=1, max_value=6, value=1, step=1)
        for i in range(int(n_stripes)):
            orient_label = ""
            if stripe_orientation == "Mix H+V":
                orient_label = " [H]" if i % 2 == 0 else " [V]"
            st.caption(f"Striscia {i+1}{orient_label}")
            col_a, col_b = st.columns(2)
            with col_a:
                pos  = st.slider(f"Posizione {i+1} (%)", 0, 95, min(20 + i*25, 90), key=f"sp_{i}",
                    help="Dove inizia (0=cima/sinistra → 100=fondo/destra)")
            with col_b:
                size = st.slider(f"Larghezza {i+1} (%)", 1, 50, 10, key=f"ss_{i}",
                    help="Quanto è larga")
            stripes.append((pos, size))

        st.divider()

        # --- ANTEPRIMA ---
        prev_choices = []
        prev_files   = {}
        if up_m1: prev_choices.append("Master 1");              prev_files["Master 1"] = up_m1
        if up_m2: prev_choices.append("Master 2");              prev_files["Master 2"] = up_m2
        if up_t:  prev_choices.append("Prima foto Calderone");  prev_files["Prima foto Calderone"] = up_t[0]

        if prev_choices:
            prev_sel = st.selectbox("🔍 Anteprima su", prev_choices)
            pf = prev_files[prev_sel]
            pf.seek(0)
            prev_img   = np.array(Image.open(pf).convert("RGB"))
            ph, pw     = prev_img.shape[:2]
            scale      = 300 / max(ph, pw)
            dw, dh     = int(pw * scale), int(ph * scale)
            prev_small = cv2.resize(prev_img, (dw, dh))
            overlay    = prev_small.copy()

            def _draw_stripe_h(ov, y0, y1, dh):
                ov[y0:y1, :] = (ov[y0:y1, :] * 0.35 + np.array([120, 80, 220]) * 0.65).astype(np.uint8)
                if y0 > 1:  ov[max(0,y0-2):y0, :] = [80, 40, 200]
                if y1 < dh: ov[y1:min(dh,y1+2), :] = [80, 40, 200]

            def _draw_stripe_v(ov, x0, x1, dw):
                ov[:, x0:x1] = (ov[:, x0:x1] * 0.35 + np.array([120, 80, 220]) * 0.65).astype(np.uint8)
                if x0 > 1:  ov[:, max(0,x0-2):x0] = [80, 40, 200]
                if x1 < dw: ov[:, x1:min(dw,x1+2)] = [80, 40, 200]

            for idx, (pos, size) in enumerate(stripes):
                use_h = (stripe_orientation == "Orizzontale") or \
                        (stripe_orientation == "Mix H+V" and idx % 2 == 0)
                if use_h:
                    y0 = int(np.clip(pos / 100.0, 0, 1) * dh)
                    y1 = int(np.clip((pos + size) / 100.0, 0, 1) * dh)
                    _draw_stripe_h(overlay, y0, y1, dh)
                else:
                    x0 = int(np.clip(pos / 100.0, 0, 1) * dw)
                    x1 = int(np.clip((pos + size) / 100.0, 0, 1) * dw)
                    _draw_stripe_v(overlay, x0, x1, dw)

            caption = "Anteprima — zona viola = striscia attiva"
            if stripe_reverse:
                caption += " · REVERSE attivo (strisce = Master fermo)"
            st.image(overlay, caption=caption, use_container_width=False)
        else:
            st.caption("Carica almeno una foto per vedere l'anteprima.")

    st.divider()
    seq_mode = st.toggle("🔢 Sequenza Ordinata", value=False,
        help="Le foto del Calderone vengono usate in ordine (1→2→3…) invece che random.")

with c3:
    st.subheader("🎬 Rendering")
    fmt = st.selectbox("Format", ["16:9 (Orizzontale)", "9:16 (Verticale)", "1:1 (Quadrato)"])
    dur = st.number_input("Durata (sec)", 1, 300, 10)

    st.divider()

    beat_sync = st.toggle("🎵 A tempo di musica", value=False,
        help="Attiva solo se hai caricato un audio. Disabilitato in modalità Slideshow.")
    genre = "Techno / House"
    if beat_sync:
        genre = st.selectbox("Genere", list(GENRE_PRESETS.keys()))

    st.divider()

    slideshow_mode = st.toggle("📽️ Modalità Slideshow", value=False,
        help="Sfoglia le foto lentamente con transizioni glitch. Il ritmo è controllato dagli slider, non dal beat.")
    slide_hold       = 3.0
    slide_trans      = 2.0
    slide_trans_type = "Dissolve Glitchato"
    if slideshow_mode:
        slide_hold       = st.slider("🖼️ Durata foto (sec)",        1.0, 15.0, 3.0, step=0.5)
        slide_trans      = st.slider("🌀 Durata transizione (sec)",  0.5, 10.0, 2.0, step=0.5)
        slide_trans_type = st.radio("Tipo transizione",
            ["Dissolve Glitchato", "Glitch Burst"],
            help="Dissolve: le foto si mescolano con glitch. Burst: foto ferma → esplode → foto nuova.")

    st.divider()

    if st.button("🚀 AVVIA DISSEZIONE"):
        v, r, base_name = generate_master(
            up_m1, up_m2, up_t, up_a,
            mode, lines, dur,
            chaos, speed, fmt,
            m1_end_t, m2_start_t,
            rand_l,
            beat_sync, genre,
            seq_mode,
            slideshow_mode, slide_hold, slide_trans, slide_trans_type,
            stripe_mode, stripes, stripe_orientation,
            stripe_bg, stripe_glitch, stripe_reverse
        )
        st.session_state.v_path  = v
        st.session_state.r_path  = r
        st.session_state.file_ts = base_name

    if st.session_state.v_path:
        st.video(st.session_state.v_path)
        base = st.session_state.file_ts or "video_dissection"
        st.download_button("💾 DOWNLOAD VIDEO",
            open(st.session_state.v_path, "rb"),
            file_name=f"{base}.mp4")
        if st.session_state.r_path:
            with open(st.session_state.r_path, "r") as f: r_txt = f.read()
            st.text_area("📄 TECHNICAL REPORT", r_txt, height=380)
            st.download_button("📄 SCARICA REPORT", r_txt,
                file_name=f"{base}_report.txt")
