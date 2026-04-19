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

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Recursive-Cut-Photo by Loop507", layout="wide")

if 'v_path' not in st.session_state: st.session_state.v_path = None
if 'r_path' not in st.session_state: st.session_state.r_path = None

# --- PRESET GENERE ---
GENRE_PRESETS = {
    "Techno / House":  {"beat_strength": 70, "beat_decay": 20, "onset": 0,  "cache": 20},
    "Orchestrale":     {"beat_strength": 30, "beat_decay": 80, "onset": 40, "cache": 25},
    "Pop / Soul":      {"beat_strength": 50, "beat_decay": 50, "onset": 60, "cache": 35},
    "Glitch / Noise":  {"beat_strength": 90, "beat_decay": 10, "onset": 80, "cache": 70},
    "Drone / Pad":     {"beat_strength":  0, "beat_decay": 60, "onset": 20, "cache": 15},
    "Hip Hop / Jazz":  {"beat_strength": 60, "beat_decay": 35, "onset": 50, "cache": 30},
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

def generate_master(up_m1, up_m2, up_trit, up_aud,
                    orientation, strand_val, max_limit,
                    chaos_val, photo_speed, format_type,
                    start_c, end_c,
                    rand_lines,
                    beat_sync, genre,
                    rhythm_tracking):

    fps = 24
    total_f = int(max_limit * fps)
    prog_bar = st.progress(0)

    # --- ASSETS — riduzione a metà risoluzione durante generazione ---
    def load_img(f): return resize_to_format(np.array(Image.open(f).convert("RGB")), format_type, half_res=True)

    img_m1 = load_img(up_m1) if up_m1 else None
    img_m2 = load_img(up_m2) if up_m2 else None
    pool_imgs = [load_img(f) for f in up_trit] if up_trit else []
    if not pool_imgs:
        pool_imgs = [np.zeros((360, 640, 3), dtype=np.uint8)]

    h, w = pool_imgs[0].shape[:2]

    # Output a risoluzione piena — upscale solo in scrittura finale
    if format_type == "16:9 (Orizzontale)": out_w, out_h = 1280, 720
    elif format_type == "9:16 (Verticale)":  out_w, out_h = 720, 1280
    else:                                     out_w, out_h = 1080, 1080

    # --- AUDIO ANALYSIS ---
    audio_envelope  = np.ones(total_f)
    beat_envelope   = np.zeros(total_f)
    onset_envelope  = np.zeros(total_f)
    rhythm_envelope = None
    audio_peak      = 0.0
    beat_count      = 0
    temp_aud_path   = None

    if up_aud:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as t:
            t.write(up_aud.read())
            temp_aud_path = t.name

        y, sr = librosa.load(temp_aud_path, sr=22050, mono=True, duration=max_limit)

        # RMS
        rms = librosa.feature.rms(y=y)[0]
        audio_peak = float(np.max(rms))
        audio_envelope = np.interp(
            np.linspace(0, len(rms)-1, total_f),
            np.arange(len(rms)), rms / (rms.max() + 1e-6)
        )

        # Beat + Onset dal preset genere
        if beat_sync:
            p = GENRE_PRESETS[genre]
            bs = p["beat_strength"]
            bd = p["beat_decay"]
            op = p["onset"]
            bc = p["cache"]

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
        else:
            bs, bd, op, bc = 0, 50, 0, 30

        # Rhythm Tracking
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

    # --- CACHE FOTO con limite massimo ---
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

    # --- POWER CURVE semplificata: chaos controlla solo intensità strisce ---
    c_n = chaos_val / 100.0
    sv = int(85 * (1 - c_n) + 2)
    pv = min(int(100 * (1 - c_n) + 5), 100)
    ev = int(75 * (1 - c_n) + 2)

    def make_frame(t):
        f = int(t * fps)
        if f >= total_f: f = total_f - 1
        prog_bar.progress(f / total_f)
        prog = t / max_limit

        # --- SELEZIONE FOTO: logica transizione M1 → Calderone → M2 ---
        has_masters = (img_m1 is not None) and (img_m2 is not None)

        if has_masters:
            if prog < start_c:
                # Solo M1
                def pick():
                    key = f // max(1, int(fps / photo_speed))
                    if key in cached_picks and random.random() > 0.1:
                        return cached_picks[key]
                    cache_set(key, img_m1)
                    return img_m1
            elif prog > end_c:
                # Solo M2
                def pick():
                    key = f // max(1, int(fps / photo_speed))
                    if key in cached_picks and random.random() > 0.1:
                        return cached_picks[key]
                    cache_set(key, img_m2)
                    return img_m2
            else:
                # Transizione: mix M1, Calderone, M2 in base a chaos e posizione
                t_rel = (prog - start_c) / (end_c - start_c)
                def pick():
                    interval = max(1, int(fps / photo_speed))
                    key = f // interval
                    force = onset_envelope[f] > 0 and random.random() < (bc / 100.0) if beat_sync else False
                    if key in cached_picks and not force and random.random() > 0.1:
                        return cached_picks[key]
                    r = random.random()
                    chaos_prob = c_n * 0.6
                    m1_prob = (1 - t_rel) * (1 - chaos_prob)
                    m2_prob = t_rel * (1 - chaos_prob)
                    if r < m1_prob: res = img_m1
                    elif r < m1_prob + m2_prob: res = img_m2
                    else: res = random.choice(pool_imgs)
                    cache_set(key, res)
                    return res
        else:
            # Solo Calderone
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
                res = random.choice(pool_imgs)
                cache_set(key, res)
                return res

        # --- VAL: rhythm bypassa power curve se attivo ---
        if rhythm_envelope is not None:
            val = rhythm_envelope[f]
        else:
            mid = 0.5
            v_base = (sv + (prog/mid)*(pv-sv)) if prog <= mid else (pv + ((prog-mid)/mid)*(ev-pv))
            val = (v_base / 100.0) * audio_envelope[f]

        if beat_sync and beat_envelope[f] > 0:
            val = val * (1.0 + beat_envelope[f] * (bs / 100.0))

        # --- SHIFT con strand adattato alla mezza risoluzione ---
        strand = max(1, strand_val // 2)

        def get_b(max_d):
            res, c = [], 0
            while c < max_d:
                sw = random.randint(max(2, int(strand*0.6)), int(strand*1.4)) if rand_lines else strand
                res.append((c, min(c+sw, max_d)))
                c += sw
            return res

        if orientation == "Nessun Effetto":
            return cv2.resize(pick(), (out_w, out_h))

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
            # OTTIMIZZATO: roll sull'intero frame, poi patch — non patch per patch
            target = pick()
            shift_h = int(random.uniform(-250, 250) * val)
            shift_v = int(random.uniform(-250, 250) * val)
            rolled = np.roll(np.roll(target, shift_h, axis=1), shift_v, axis=0)
            bh_list = get_b(h)
            bw_list = get_b(w)
            for bh in bh_list:
                for bw in bw_list:
                    # Alterna tra frame originale e rolled per effetto mosaico
                    if random.random() > 0.5:
                        frame[bh[0]:bh[1], bw[0]:bw[1]] = rolled[bh[0]:bh[1], bw[0]:bw[1]]
                    else:
                        frame[bh[0]:bh[1], bw[0]:bw[1]] = target[bh[0]:bh[1], bw[0]:bw[1]]

        # Upscale a risoluzione finale
        return cv2.resize(frame, (out_w, out_h))

    clip = VideoClip(make_frame, duration=max_limit)
    if temp_aud_path:
        audio_clip = AudioFileClip(temp_aud_path)
        if audio_clip.duration < max_limit: audio_clip = audio_loop(audio_clip, duration=max_limit)
        else: audio_clip = audio_clip.set_duration(max_limit)
        clip = clip.set_audio(audio_clip)

    v_out = tempfile.mktemp(suffix=".mp4")
    clip.write_videofile(v_out, codec="libx264", audio_codec="aac" if up_aud else None,
                         fps=fps, bitrate="5000k", logger=None)

    report_text = f"""[SLICE_PHOTO_DISSECTION] // VOL_01 // H.264 // DATA_FRAGMENT

:: MOTORE: recursive_cut_pro [v9.0]
:: EFFETTO: Recursive Strand Shift
:: ANALISI: RMS / Beat Sync / Rhythm Tracking

---
> TECHNICAL LOG SHEET:
* Asset Pool: {len(pool_imgs)} foto
* Rendering: {total_f} frame @ {fps}fps
* Geometria: {orientation} @ {strand_val}px
* Chaos: {chaos_val}% | Photo Speed: {photo_speed}fps
* Transizione: {int(start_c*100)}% — {int(end_c*100)}%
* Audio Peak: {audio_peak:.4f}
* Beat Sync: {'ON — ' + genre + ' — ' + str(beat_count) + ' beat' if beat_sync else 'OFF'}
* Rhythm Tracking: {'ON' if rhythm_tracking else 'OFF'}
* Power Curve: {'BYPASSED' if rhythm_tracking else 'ON'}

> Regia e Algoritmo: Loop507

#Loop507 #SlicePhoto #StrandShift #GlitchArt #ExperimentalVideo"""

    r_out = tempfile.mktemp(suffix=".txt")
    with open(r_out, "w") as f: f.write(report_text)
    if temp_aud_path: os.remove(temp_aud_path)
    gc.collect()
    return v_out, r_out

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

    # Transizione — compare solo se entrambi i master sono caricati
    has_masters = (up_m1 is not None) and (up_m2 is not None)
    if has_masters:
        st.caption("Transizione M1 → Calderone → M2")
        start_t = st.slider("Inizio transizione (%)", 0, 50, 10) / 100.0
        end_t   = st.slider("Fine transizione (%)",  50, 100, 90) / 100.0
        st.divider()
    else:
        start_t, end_t = 0.1, 0.9
        if up_m1 or up_m2:
            st.caption("⚠️ Carica entrambi i Master per attivare la transizione.")

    chaos  = st.slider("🌀 Chaos", 0, 100, 50)
    speed  = st.slider("⚡ Photo Speed (fps)", 1, 24, 6)
    lines  = st.slider("📐 Strand (px)", 1, 500, 45)
    rand_l = st.toggle("Dynamic Slicing", value=True)
    mode   = st.radio("Geometria", ["Orizzontale", "Verticale", "Mix (H+V)", "Mosaico", "Nessun Effetto"])

with c3:
    st.subheader("🎬 Rendering")
    fmt = st.selectbox("Format", ["16:9 (Orizzontale)", "9:16 (Verticale)", "1:1 (Quadrato)"])
    dur = st.number_input("Durata (sec)", 1, 300, 10)

    st.divider()

    # Sync audio — toggle + preset genere
    beat_sync = st.toggle("🥁 A tempo di musica", value=False,
        help="Attiva solo se hai caricato un audio.")
    genre = "Techno / House"
    if beat_sync:
        genre = st.selectbox("Genere", list(GENRE_PRESETS.keys()))

    rhythm_tracking = st.toggle("🎼 Segui il ritmo del brano", value=False,
        help="Bypassa la Power Curve. Le strisce rallentano e accelerano con il brano.")

    st.divider()

    if st.button("🚀 AVVIA DISSEZIONE"):
        v, r = generate_master(
            up_m1, up_m2, up_t, up_a,
            mode, lines, dur,
            chaos, speed, fmt,
            start_t, end_t,
            rand_l,
            beat_sync, genre,
            rhythm_tracking
        )
        st.session_state.v_path, st.session_state.r_path = v, r

    if st.session_state.v_path:
        st.video(st.session_state.v_path)
        st.download_button("💾 DOWNLOAD VIDEO",
            open(st.session_state.v_path, "rb"), "video_dissection.mp4")
        if st.session_state.r_path:
            with open(st.session_state.r_path, "r") as f: r_txt = f.read()
            st.text_area("📄 TECHNICAL REPORT", r_txt, height=380)
            st.download_button("📄 SCARICA REPORT", r_txt, "report_dissection.txt")
