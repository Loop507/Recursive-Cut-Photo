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

def generate_master(up_m1, up_m2, up_trit, up_aud, orientation, strand_val, max_limit, k_p, m1_s, m1_e, m2_s, m2_e, p_m1_ctrl, p_m2_ctrl, start_c, end_c, format_type, inc_master, rand_lines, photo_speed, chaos_val,
                    beat_strength, onset_photo_switch, beat_decay, beat_cache_sensitivity, rhythm_tracking):
    fps = 24
    total_f = int(max_limit * fps)
    prog_bar = st.progress(0)

    # Assets
    img_m1 = resize_to_format(np.array(Image.open(up_m1).convert("RGB")), format_type) if up_m1 else None
    img_m2 = resize_to_format(np.array(Image.open(up_m2).convert("RGB")), format_type) if up_m2 else None
    t_processed = [resize_to_format(np.array(Image.open(f).convert("RGB")), format_type) for f in up_trit] if up_trit else []
    
    pool_imgs = t_processed.copy()
    if inc_master:
        if img_m1 is not None: pool_imgs.append(img_m1)
        if img_m2 is not None: pool_imgs.append(img_m2)
    if not pool_imgs: pool_imgs = [np.zeros((720, 1280, 3), dtype=np.uint8)]
    
    h, w = pool_imgs[0].shape[:2]
    
    # --- AUDIO ANALYSIS ---
    audio_envelope = np.ones(total_f)
    beat_envelope = np.zeros(total_f)
    onset_envelope = np.zeros(total_f)
    rhythm_envelope = None  # None = non attivo, array = attivo e comanda val
    audio_peak = 0.0
    beat_count = 0
    temp_aud_path = None

    if up_aud:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as t_aud:
            t_aud.write(up_aud.read())
            temp_aud_path = t_aud.name
        y, sr = librosa.load(temp_aud_path, sr=22050, mono=True, duration=max_limit)

        # RMS — sempre calcolato, usato solo se rhythm_tracking è OFF
        rms = librosa.feature.rms(y=y)[0]
        audio_peak = float(np.max(rms))
        audio_envelope = np.interp(
            np.linspace(0, len(rms)-1, total_f),
            np.arange(len(rms)), rms / (rms.max() + 1e-6)
        )

        # Beat + Onset
        if beat_strength > 0:
            _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            beat_count = len(beat_times)
            for bt in beat_times:
                bf = int(bt * fps)
                decay_rate = 1.0 - (beat_decay / 100.0) * 0.98
                for df in range(min(int(fps * 0.5), total_f - bf)):
                    beat_envelope[bf + df] = max(beat_envelope[bf + df], decay_rate ** df)

        if onset_photo_switch > 0:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            for ot in onset_times:
                of = int(ot * fps)
                if of < total_f:
                    onset_envelope[of] = 1.0

        # Rhythm Tracking — quando attivo SOSTITUISCE la power curve
        if rhythm_tracking:
            # Tempogram: velocità ritmica locale
            tempogram = librosa.feature.tempogram(y=y, sr=sr)
            tempo_local = tempogram.max(axis=0).astype(float)
            tempo_local = tempo_local / (tempo_local.max() + 1e-6)

            # Spectral Flux: cambi di timbro/attacco
            stft = np.abs(librosa.stft(y))
            flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
            flux = flux / (flux.max() + 1e-6)

            # Combinazione 50/50
            min_len = min(len(tempo_local), len(flux))
            combined = (tempo_local[:min_len] * 0.5 + flux[:min_len] * 0.5)
            combined = combined / (combined.max() + 1e-6)

            # Interpolazione a total_f frame
            rhythm_envelope = np.interp(
                np.linspace(0, len(combined)-1, total_f),
                np.arange(len(combined)), combined
            )

            # Smoothing — finestra mezza secondo
            kernel = np.ones(fps // 2) / (fps // 2)
            rhythm_envelope = np.convolve(rhythm_envelope, kernel, mode='same')

            # Normalizza tra 0 e 1 — nessun floor artificiale:
            # su brani lenti deve poter andare davvero vicino a zero
            rhythm_envelope = rhythm_envelope / (rhythm_envelope.max() + 1e-6)
            rhythm_envelope = np.clip(rhythm_envelope, 0.0, 1.0)

    cached_picks = {}

    def make_frame(t):
        f = int(t * fps)
        if f >= total_f: f = total_f - 1
        prog_bar.progress(f / total_f)
        prog = t / max_limit

        # Transizione statistica — invariata
        if prog < start_c:
            prob_m1, prob_m2, prob_c = 1.0, 0.0, 0.0
        elif prog > end_c:
            prob_m1, prob_m2, prob_c = 0.0, 1.0, 0.0
        else:
            t_rel = (prog - start_c) / (end_c - start_c)
            prob_m1 = (1.0 - t_rel) * (p_m1_ctrl / 100.0)
            prob_m2 = t_rel * (p_m2_ctrl / 100.0)
            prob_c = (chaos_val / 100.0)
            total_p = prob_m1 + prob_m2 + prob_c + 1e-6
            prob_m1, prob_m2, prob_c = prob_m1/total_p, prob_m2/total_p, prob_c/total_p

        # --- VAL: logica biforcata ---
        if rhythm_envelope is not None:
            # RHYTHM ON: rhythm_envelope comanda direttamente — power curve bypassata
            val = rhythm_envelope[f]
        else:
            # RHYTHM OFF: comportamento originale invariato
            mid = 0.5
            v_base = (k_p['sv'] + (prog/mid)*(k_p['pv']-k_p['sv'])) if prog <= mid else \
                     (k_p['pv'] + ((prog-mid)/mid)*(k_p['ev']-k_p['pv']))
            val = (v_base / 100.0) * audio_envelope[f]

        # Beat amplifica val in entrambe le modalità
        if beat_strength > 0:
            val = val * (1.0 + beat_envelope[f] * (beat_strength / 100.0))

        mag1 = (m1_s + prog * (m1_e - m1_s)) / 100
        mag2 = (m2_s + prog * (m2_e - m2_s)) / 100
        dist_mult = 1.0 - np.clip(mag1 + mag2, 0, 0.95)

        def pick():
            # Photo speed modulata dal ritmo se attivo
            if rhythm_envelope is not None:
                speed_mod = max(1, photo_speed * (0.2 + rhythm_envelope[f] * 0.8))
            else:
                speed_mod = photo_speed
            interval = max(1, int(fps / speed_mod))
            key = f // interval
            force_change = (
                onset_envelope[f] > 0 and
                random.random() < (onset_photo_switch / 100.0) and
                random.random() < (beat_cache_sensitivity / 100.0)
            )
            if key in cached_picks and not force_change and random.random() > 0.1:
                return cached_picks[key]
            r = random.random()
            if img_m1 is not None and r < prob_m1: res = img_m1
            elif img_m2 is not None and r < (prob_m1 + prob_m2): res = img_m2
            else: res = random.choice(pool_imgs)
            cached_picks[key] = res
            return res

        if orientation == "Nessun Effetto": return pick()

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        def get_b(max_d):
            res, c = [], 0
            while c < max_d:
                sw = random.randint(max(2, int(strand_val*0.6)), int(strand_val*1.4)) if rand_lines else strand_val
                res.append((c, min(c+sw, max_d)))
                c += sw
            return res

        if orientation == "Orizzontale":
            for s, e in get_b(h):
                target = pick(); shift = int(random.uniform(-500, 500) * val * dist_mult)
                frame[s:e, :] = np.roll(target[s:e, :], shift, axis=1)
        elif orientation == "Verticale":
            for s, e in get_b(w):
                target = pick(); shift = int(random.uniform(-500, 500) * val * dist_mult)
                frame[:, s:e] = np.roll(target[:, s:e], shift, axis=0)
        elif orientation == "Mix (H+V)":
            for s, e in get_b(h):
                target = pick(); shift = int(random.uniform(-500, 500) * val * dist_mult)
                frame[s:e, :] = np.roll(target[s:e, :], shift, axis=1)
            for s, e in get_b(w):
                if random.random() > 0.5:
                    shift_v = int(random.uniform(-400, 400) * val * dist_mult)
                    frame[:, s:e] = np.roll(frame[:, s:e], shift_v, axis=0)
        elif orientation == "Mosaico":
            blocks_h = get_b(h)
            blocks_w = get_b(w)
            for bh in blocks_h:
                for bw in blocks_w:
                    target = pick()
                    shift = int(random.uniform(-400, 400) * val * dist_mult)
                    axis = random.choice([0, 1])
                    patch = target[bh[0]:bh[1], bw[0]:bw[1]]
                    frame[bh[0]:bh[1], bw[0]:bw[1]] = np.roll(patch, shift, axis=axis)
        return frame

    clip = VideoClip(make_frame, duration=max_limit)
    if temp_aud_path:
        audio_clip = AudioFileClip(temp_aud_path)
        if audio_clip.duration < max_limit: audio_clip = audio_loop(audio_clip, duration=max_limit)
        else: audio_clip = audio_clip.set_duration(max_limit)
        clip = clip.set_audio(audio_clip)

    v_out = tempfile.mktemp(suffix=".mp4")
    clip.write_videofile(v_out, codec="libx264", audio_codec="aac" if up_aud else None, fps=fps, bitrate="5000k", logger=None)

    report_text = f"""[SLICE_PHOTO_DISSECTION] // VOL_01 // H.264 // DATA_FRAGMENT

:: STILE: Minimalismo Computazionale / Dissezione Brutalista
:: MOTORE: recursive_cut_pro [v8.1]
:: EFFETTO: Recursive Strand Shift (Reattivo)
:: ANALISI: RMS Signal Analysis / Beat Sync / Rhythm Tracking
:: PROCESSO: Frammentazione Ricorsiva / Magnetismo Forzato

"L'immagine è stata smontata. Il codice ne ha riscritto la struttura."

---
> TECHNICAL LOG SHEET:
* Asset Pool: {len(pool_imgs)} foto dissezionate
* Rendering: {total_f} frame totali generati
* Geometria: {orientation} @ {strand_val}px
* Power Curve: {'BYPASSED — Rhythm Tracking attivo' if rhythm_tracking else f"Start {k_p['sv']}% | Peak {k_p['pv']}% | End {k_p['ev']}%"}
* Magnetismo: Inizio Snap @ {max_limit * start_c:.1f}s (Pull {m1_s}%)
* Audio Peak: {audio_peak:.4f} normalized
* Beat rilevati: {beat_count} | Beat Strength: {beat_strength}% | Beat Decay: {beat_decay}%
* Onset Photo Switch: {onset_photo_switch}% | Cache Sensitivity: {beat_cache_sensitivity}%
* Rhythm Tracking: {'ON — Tempogram + Spectral Flux (Power Curve bypassata)' if rhythm_tracking else 'OFF'}

> Regia e Algoritmo: Loop507

#Loop507 #SlicePhoto #StrandShift #DigitalAnatomy #SignalCorruption #BrutalistArt 
#ComputationalMinimalism #DataDestruction #ExperimentalVideo #GlitchArt"""

    r_out = tempfile.mktemp(suffix=".txt")
    with open(r_out, "w") as f: f.write(report_text)
    if temp_aud_path: os.remove(temp_aud_path)
    gc.collect()
    return v_out, r_out

# --- INTERFACCIA ---
st.title("Recursive-Cut-Photo by Loop507 🔪")
c1, c2, c3 = st.columns([1, 1.2, 1])

with c1:
    st.subheader("🖼️ Assets")
    up_m1 = st.file_uploader("MASTER 1 (Start)", type=["jpg","png","jpeg"])
    m1_s = st.slider("M1 Start Magnetism", 0, 100, 100)
    m1_e = st.slider("M1 End Magnetism", 0, 100, 0)
    st.divider()
    up_m2 = st.file_uploader("MASTER 2 (End)", type=["jpg","png","jpeg"])
    m2_s = st.slider("M2 Start Magnetism", 0, 100, 0)
    m2_e = st.slider("M2 End Magnetism", 0, 100, 100)
    st.divider()
    up_t = st.file_uploader("CALDERONE", type=["jpg","png","jpeg"], accept_multiple_files=True)
    up_a = st.file_uploader("AUDIO", type=["mp3","wav"])
    inc_m = st.toggle("Includi Master nel Calderone", value=True)

with c2:
    st.subheader("✂️ Controllo Frame & Transizione")
    p_m1 = st.slider("Percentuale Frame M1 (Presenza)", 0, 100, 100)
    p_m2 = st.slider("Percentuale Frame M2 (Presenza)", 0, 100, 100)
    st.divider()
    start_t = st.slider("Inizio Dissoluzione (%)", 0, 50, 10) / 100.0
    end_t = st.slider("Fine Ricomposizione (%)", 50, 100, 90) / 100.0
    st.divider()
    chaos = st.slider("🌀 Chaos level", 0, 100, 50)
    c_n = chaos / 100.0
    k_params = {'sv': int(85*(1-c_n)+2), 'pv': min(int(100*(1-c_n)+5),100), 'ev': int(75*(1-c_n)+2)}
    speed = st.slider("Photo Speed (fps)", 1, 24, 6)
    lines = st.slider("Strand (px)", 1, 500, 45)
    rand_l = st.toggle("Dynamic Slicing", value=True)
    mode = st.radio("Geometria", ["Orizzontale", "Verticale", "Mix (H+V)", "Mosaico", "Nessun Effetto"])

with c3:
    st.subheader("🎬 Rendering")
    fmt = st.selectbox("Format", ["16:9 (Orizzontale)", "9:16 (Verticale)", "1:1 (Quadrato)"])
    dur = st.number_input("Durata (sec)", 1, 120, 10)

    st.divider()
    st.subheader("🥁 Sync Audio Ritmico")
    st.caption("Tutti a zero = comportamento identico all'originale")
    beat_strength = st.slider("Beat Strength", 0, 100, 0)
    beat_decay = st.slider("Beat Decay", 0, 100, 50)
    onset_photo_switch = st.slider("Onset Photo Switch", 0, 100, 0)
    beat_cache_sensitivity = st.slider("Beat Cache Sensitivity", 0, 100, 30)
    rhythm_tracking = st.toggle(
        "🎼 Segui il Ritmo del Brano",
        value=False,
        help="Bypassa la Power Curve: sono il Tempogram e lo Spectral Flux a guidare le strisce. Su brani lenti le strisce rallentano davvero. Su brani veloci esplodono."
    )
    st.divider()

    if st.button("🚀 AVVIA DISSEZIONE"):
        v, r = generate_master(
            up_m1, up_m2, up_t, up_a, mode, lines, dur, k_params,
            m1_s, m1_e, m2_s, m2_e, p_m1, p_m2, start_t, end_t,
            fmt, inc_m, rand_l, speed, chaos,
            beat_strength, onset_photo_switch, beat_decay, beat_cache_sensitivity, rhythm_tracking
        )
        st.session_state.v_path, st.session_state.r_path = v, r

    if st.session_state.v_path:
        st.video(st.session_state.v_path)
        st.download_button("💾 DOWNLOAD VIDEO", open(st.session_state.v_path, "rb"), "video_dissection.mp4")
        if st.session_state.r_path:
            with open(st.session_state.r_path, "r") as f: r_txt = f.read()
            st.text_area("📄 TECHNICAL REPORT", r_txt, height=480)
            st.download_button("📄 SCARICA REPORT", r_txt, "report_dissection.txt")
