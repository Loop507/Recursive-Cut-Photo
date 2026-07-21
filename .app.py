import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from moviepy.editor import VideoClip, AudioFileClip
from moviepy.audio.fx.all import audio_loop
import tempfile
import random
import os
import librosa
import gc
import json
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Recursive-Cut-Photo by Loop507", layout="wide")

if 'v_path'       not in st.session_state: st.session_state.v_path       = None
if 'r_path'       not in st.session_state: st.session_state.r_path       = None
if 'file_ts'      not in st.session_state: st.session_state.file_ts      = None
if 'preset'       not in st.session_state: st.session_state.preset       = None
if 'frame_export' not in st.session_state: st.session_state.frame_export = None
if 'stripe_ids'   not in st.session_state: st.session_state.stripe_ids   = [0]
if 'stripe_next_id' not in st.session_state: st.session_state.stripe_next_id = 1
if 'overlay_ids'     not in st.session_state: st.session_state.overlay_ids     = []
if 'overlay_next_id' not in st.session_state: st.session_state.overlay_next_id = 0

# --- PRESET GENERE ---
GENRE_PRESETS = {
    "Techno / House":  {"beat_strength": 70, "beat_decay": 20, "onset":  0, "cache": 20, "rhythm": False},
    "Orchestrale":     {"beat_strength": 20, "beat_decay": 80, "onset": 40, "cache": 25, "rhythm": True},
    "Pop / Soul":      {"beat_strength": 50, "beat_decay": 50, "onset": 60, "cache": 35, "rhythm": False},
    "Glitch / Noise":  {"beat_strength": 90, "beat_decay": 10, "onset": 80, "cache": 70, "rhythm": True},
    "Drone / Pad":     {"beat_strength":  0, "beat_decay": 60, "onset": 20, "cache": 15, "rhythm": True},
    "Hip Hop / Jazz":  {"beat_strength": 60, "beat_decay": 35, "onset": 50, "cache": 30, "rhythm": False},
}

# Config condivisa per le strisce con moto "a rampa" (angolo/raggio che cresce nel tempo,
# eventualmente accelerando sul beat). Orizzontale/Verticale non è qui perché il suo moto automatico è
# un'oscillazione sinusoidale, non una rampa, e vive nel proprio ramo in generate_master.
STRIPE_MOTION_CONFIG = {
    "Lancetta":         dict(auto_key="auto_rotate", speed_key="rotate_speed", speed_default=30.0,
                             base_key="angle", base_default=90.0, wrap_hi=360.0,
                             add_base_to_ramp=True),
    "Striscia Ruotata": dict(auto_key="auto_rotate", speed_key="rotate_speed", speed_default=30.0,
                             base_key="angle", base_default=0.0, wrap_hi=360.0,
                             add_base_to_ramp=True),
    "Cerchio":          dict(auto_key="auto_expand", speed_key="expand_speed", speed_default=20.0,
                             base_key="radius", base_default=30.0, wrap_hi=100.0,
                             add_base_to_ramp=False),
}


def resolve_reactive_val(s_dict, base_val, offset, auto_key):
    """
    Sceglie tra valore statico/keyframe (base_val) e offset animato (rampa continua),
    a seconda che 'auto_key' (es. auto_rotate/auto_expand/move_random) sia attivo per
    questa striscia. 'beat_react' da solo non tocca la posizione: con solo
    'Sincronizza al beat' attivo (senza il moto automatico associato), la striscia
    resta ferma al suo valore statico/keyframe — niente scatti né attenuazioni
    automatiche d'opacità.
    """
    if s_dict.get(auto_key, False):
        return offset
    return base_val

# =====================================================================
# KEYFRAME INTERPOLATION
# =====================================================================

def kf_interp(kf_list, t, total_dur):
    """
    Interpola linearmente una lista di keyframe [{t, v}, ...] al tempo t (sec).
    Se la lista è vuota o None, ritorna None (usa valore statico).
    kf_list è ordinata per t crescente.
    """
    if not kf_list or len(kf_list) == 0:
        return None
    kf_sorted = sorted(kf_list, key=lambda k: k['t'])
    if t <= kf_sorted[0]['t']:
        return float(kf_sorted[0]['v'])
    if t >= kf_sorted[-1]['t']:
        return float(kf_sorted[-1]['v'])
    for a, b in zip(kf_sorted, kf_sorted[1:]):
        if a['t'] <= t <= b['t']:
            span = b['t'] - a['t']
            if span < 1e-6:
                return float(b['v'])
            alpha = (t - a['t']) / span
            return float(a['v'] + (b['v'] - a['v']) * alpha)
    return float(kf_sorted[-1]['v'])


def kf_get(s_dict, param, t, total_dur, default):
    """
    Restituisce il valore animato di `param` per la striscia s_dict al tempo t.
    Se non ci sono KF per quel parametro, usa il valore statico.
    """
    kf_list = s_dict.get('keyframes', {}).get(param, [])
    val = kf_interp(kf_list, t, total_dur)
    if val is None:
        return s_dict.get(param, default)
    return val


def kf_expander_ui(stripe_id, dur, params_meta, prefix="stripe", label="Keyframe striscia"):
    """
    Un solo expander per striscia (o livello) — tabella keyframe unificata.
    params_meta: lista di dict { param, label, min_v, max_v, default_v, step_v }
    Ritorna dict { param: [{t, v}, ...] }
    'prefix' distingue lo state_key (es. 'stripe' vs 'layer') per evitare collisioni
    quando strisce e livelli condividono lo stesso indice numerico.
    """
    state_key = f"kf_{prefix}_{stripe_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {}
    kf_state = st.session_state[state_key]
    for m in params_meta:
        kf_state.setdefault(m['param'], [])

    param_labels = [m['label'] for m in params_meta]
    param_map    = {m['param']: m for m in params_meta}

    total_kf = sum(len(v) for v in kf_state.values())
    exp_label = f"\U0001f3ac {label} {stripe_id+1}" + (f"  \u2014 {total_kf} KF attivi" if total_kf else "")

    with st.expander(exp_label, expanded=False):
        st.caption("Imposta i valori con gli slider sopra, poi aggiungi qui i cambiamenti nel tempo.")
        ca, cb, cc, cd = st.columns([2, 2, 2, 1])
        with ca:
            sel_label = st.selectbox("Parametro", param_labels, key=f"kf_sel_{prefix}_{stripe_id}")
            sel_param = next(m['param'] for m in params_meta if m['label'] == sel_label)
            meta = param_map[sel_param]
        with cb:
            new_t = st.number_input("t (sec)", min_value=0.0, max_value=float(max(dur, 1)),
                                    value=0.0, step=0.5, key=f"kf_t_{prefix}_{stripe_id}")
        with cc:
            new_v = st.number_input("valore", min_value=float(meta['min_v']),
                                    max_value=float(meta['max_v']),
                                    value=float(meta['default_v']),
                                    step=float(meta.get('step_v', 1)),
                                    key=f"kf_v_{prefix}_{stripe_id}")
        with cd:
            st.write("")
            st.write("")
            if st.button("\u2795 Aggiungi", key=f"kf_add_{prefix}_{stripe_id}"):
                kf_state[sel_param].append({'t': round(float(new_t), 2), 'v': round(float(new_v), 4)})
                kf_state[sel_param].sort(key=lambda k: k['t'])
                st.session_state[state_key] = kf_state

        all_kfs = [(p, ki, kf) for p, lst in kf_state.items() for ki, kf in enumerate(lst)]
        all_kfs.sort(key=lambda x: x[2]['t'])

        if all_kfs:
            st.divider()
            to_del = None
            for p, ki, kf in all_kfs:
                lbl = param_map[p]['label']
                r1, r2, r3, r4 = st.columns([2, 3, 3, 1])
                with r1: st.caption(f"t = **{kf['t']:.1f}s**")
                with r2: st.caption(lbl)
                with r3: st.caption(f"\u2192 {kf['v']:.2f}")
                with r4:
                    if st.button("\u2715", key=f"kf_del_{prefix}_{stripe_id}_{p}_{ki}"):
                        to_del = (p, ki)
            if to_del:
                p, ki = to_del
                kf_state[p].pop(ki)
                st.session_state[state_key] = kf_state
        else:
            st.caption("\u2014 Nessun KF: si usano i valori degli slider per tutta la durata.")

    return kf_state


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


def compute_stripe_coords(center_pct, size_pct, length_pct, offset_pct, dim):
    """
    Calcola (pos0, pos1, len0, len1) per una striscia:
    - center_pct: centro della striscia sull'asse principale (0-100)
    - size_pct:   spessore della striscia (0-100)
    - length_pct: lunghezza della striscia sull'asse secondario (0-100)
    - offset_pct: offset del centro sull'asse secondario (per movimento, 0-100)
    - dim:        (dim_principale, dim_secondaria)
    Ritorna (p0, p1, l0, l1) in pixel
    """
    dp, ds = dim
    half_size = size_pct / 2.0
    p0 = int(np.clip((center_pct - half_size) / 100.0, 0.0, 1.0) * dp)
    p1 = int(np.clip((center_pct + half_size) / 100.0, 0.0, 1.0) * dp)
    half_len = length_pct / 2.0
    l_center = offset_pct / 100.0 * ds
    l0 = int(np.clip(l_center - half_len / 100.0 * ds, 0, ds))
    l1 = int(np.clip(l_center + half_len / 100.0 * ds, 0, ds))
    return p0, p1, l0, l1


def apply_chroma(patch, amount=6):
    """Sfasa i canali R e B di ±amount pixel per chromatic aberration."""
    out = patch.copy()
    if amount < 1: return out
    out[:, :, 2] = np.roll(patch[:, :, 2],  amount, axis=1)  # R → destra
    out[:, :, 0] = np.roll(patch[:, :, 0], -amount, axis=1)  # B → sinistra
    return out


def blend_patch(base, top, mode, opacity):
    """
    Combina 'top' su 'base' secondo il blend mode, poi miscela con opacity (0-1).
    base, top: uint8 arrays stessa shape
    """
    b = base.astype(np.float32)
    t = top.astype(np.float32)

    if mode == "Normal":
        blended = t
    elif mode == "Screen":
        blended = 255.0 - (255.0 - b) * (255.0 - t) / 255.0
    elif mode == "Multiply":
        blended = (b * t) / 255.0
    elif mode == "Difference":
        blended = np.abs(b - t)
    else:
        blended = t

    result = b * (1.0 - opacity) + blended * opacity
    return np.clip(result, 0, 255).astype(np.uint8)


def cover_crop(img_rgba, target_w, target_h):
    """Ritaglia (center-crop) l'immagine per adattarla all'aspect ratio del canvas —
    stessa logica di resize_to_format usata dal Calderone: riempie tutto il formato,
    nessuna barra vuota, tagliando i bordi in eccesso invece di rimpicciolire."""
    ih, iw = img_rgba.shape[:2]
    aspect_target = target_w / target_h
    aspect_img = iw / ih
    if aspect_img > aspect_target:
        new_w = max(1, int(ih * aspect_target))
        start_x = (iw - new_w) // 2
        img_rgba = img_rgba[:, start_x:start_x + new_w]
    else:
        new_h = max(1, int(iw / aspect_target))
        start_y = (ih - new_h) // 2
        img_rgba = img_rgba[start_y:start_y + new_h, :]
    return img_rgba


def place_layer_on_canvas(layer_rgba, canvas_h, canvas_w, scale, cx_pct, cy_pct):
    """
    Ridimensiona il livello (contain-fit dentro il canvas, moltiplicato per 'scale')
    e lo posiziona al centro (cx_pct, cy_pct)% del canvas. Ritorna (rgb, alpha),
    array grandi quanto il canvas, con alpha=0 fuori dal livello.
    """
    ih, iw = layer_rgba.shape[:2]
    fit = max(min(canvas_w / iw, canvas_h / ih) * scale, 0.01)
    new_w, new_h = max(1, int(iw * fit)), max(1, int(ih * fit))
    resized = cv2.resize(layer_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cx = int(cx_pct / 100.0 * canvas_w)
    cy = int(cy_pct / 100.0 * canvas_h)
    x0, y0 = cx - new_w // 2, cy - new_h // 2

    rgb_canvas   = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    alpha_canvas = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1 = min(canvas_w, x0 + new_w)
    dst_y1 = min(canvas_h, y0 + new_h)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        rgb_canvas[dst_y0:dst_y1, dst_x0:dst_x1] = resized[src_y0:src_y1, src_x0:src_x1, :3]
        alpha_canvas[dst_y0:dst_y1, dst_x0:dst_x1, 0] = (
            resized[src_y0:src_y1, src_x0:src_x1, 3].astype(np.float32) / 255.0)

    return rgb_canvas, alpha_canvas


def blend_layer(base, top_rgb, alpha, mode):
    """
    Compone top_rgb su base secondo il blend mode, con alpha PER-PIXEL (0-1, shape h,w,1)
    che include sia il canale alpha del PNG sia l'opacita'/pulsazione del livello.
    """
    b = base.astype(np.float32)
    t = top_rgb.astype(np.float32)

    if mode == "Screen":
        blended = 255.0 - (255.0 - b) * (255.0 - t) / 255.0
    elif mode == "Multiply":
        blended = (b * t) / 255.0
    elif mode == "Difference":
        blended = np.abs(b - t)
    else:
        blended = t

    result = b * (1.0 - alpha) + blended * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def get_video_overlay_frame(cap, t):
    """
    Estrae un frame RGB dal VideoCapture 'cap' al tempo t (con loop se il video è
    più corto della durata del render). Nessun effetto/glitch: solo il fotogramma
    così com'è, che verrà poi posizionato/scalato come un overlay foto.
    """
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        return None
    dur_v = n_frames / fps_v
    t_loop = t % dur_v if dur_v > 0 else 0.0
    frame_idx = min(int(t_loop * fps_v), n_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def load_overlay_image(uploaded_file):
    """Carica una foto overlay (PNG con alpha, o JPG/JPEG) come RGBA numpy."""
    uploaded_file.seek(0)
    return np.array(Image.open(uploaded_file).convert("RGBA"))


def apply_media_overlays(frame, overlays, canvas_h, canvas_w):
    """
    Compone foto/video overlay sopra 'frame', in ordine (1 sotto ... N in cima).
    Nessun blend mode, nessuna pulsazione: solo posizione (cx, cy) e scala — come
    i controlli x/y/dimensione già usati per le strisce. Ogni overlay è un dict con
    'rgba' (frame RGBA già pronto per QUESTO istante — foto statica o frame video
    già estratto), 'scale', 'cx', 'cy'.
    """
    if not overlays:
        return frame
    out = frame
    for ov in overlays:
        rgb_c, alpha_c = place_layer_on_canvas(ov['rgba'], canvas_h, canvas_w, ov['scale'], ov['cx'], ov['cy'])
        out = blend_layer(out, rgb_c, alpha_c, "Normal")
    return out


def draw_stripe_preview_overlay(overlay, stripes, stripe_orientation, dh, dw):
    """
    Disegna sopra 'overlay' (immagine RGB dh x dw) un riquadro viola per ogni striscia
    configurata, nella posizione/dimensione impostata — usato sia dall'anteprima
    strisce "normale" sia da quella unificata coi Livelli. Muta 'overlay' in place.
    """
    def _draw_stripe_h(ov, p0, p1, l0, l1):
        p0, p1 = max(0, p0), min(dh, p1)
        l0, l1 = max(0, l0), min(dw, l1)
        if p1 > p0 and l1 > l0:
            ov[p0:p1, l0:l1] = (ov[p0:p1, l0:l1] * 0.35 + np.array([120, 80, 220]) * 0.65).astype(np.uint8)
        if p0 > 1: ov[max(0, p0 - 2):p0, l0:l1] = [80, 40, 200]
        if p1 < dh: ov[p1:min(dh, p1 + 2), l0:l1] = [80, 40, 200]

    def _draw_stripe_v(ov, p0, p1, l0, l1):
        p0, p1 = max(0, p0), min(dw, p1)
        l0, l1 = max(0, l0), min(dh, l1)
        if p1 > p0 and l1 > l0:
            ov[l0:l1, p0:p1] = (ov[l0:l1, p0:p1] * 0.35 + np.array([120, 80, 220]) * 0.65).astype(np.uint8)
        if p0 > 1: ov[l0:l1, max(0, p0 - 2):p0] = [80, 40, 200]
        if p1 < dw: ov[l0:l1, p1:min(dw, p1 + 2)] = [80, 40, 200]

    for s in stripes:
        s_orient = s.get('orientation', stripe_orientation)
        VIOLET = np.array([120, 80, 220])
        DARK_V = np.array([80, 40, 200])

        if s_orient == "Orizzontale":
            off = s.get('offset_length', 50.0)
            p0, p1, l0, l1 = compute_stripe_coords(s['center'], s['size'], s['length'], off, (dh, dw))
            _draw_stripe_h(overlay, p0, p1, l0, l1)

        elif s_orient == "Verticale":
            off = s.get('offset_length', 50.0)
            p0, p1, l0, l1 = compute_stripe_coords(s['center'], s['size'], s['length'], off, (dw, dh))
            _draw_stripe_v(overlay, p0, p1, l0, l1)

        elif s_orient == "Striscia Ruotata":
            cx_r = int(s.get("cx", 50) / 100 * dw)
            cy_r = int(s.get("cy", 50) / 100 * dh)
            angle_rad_r = np.deg2rad(s.get("angle", 0))
            cos_r, sin_r = np.cos(angle_rad_r), np.sin(angle_rad_r)
            half_h_r = max(1, int(s.get("size", 15) / 100 * max(dh, dw) / 2))
            half_w_r = max(1, int(s.get("length", 100) / 100 * max(dh, dw) / 2))
            corners_r = np.array([[-half_w_r, -half_h_r], [half_w_r, -half_h_r],
                                   [half_w_r, half_h_r], [-half_w_r, half_h_r]], dtype=np.float32)
            rot_r = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            pts_r = ((rot_r @ corners_r.T).T + np.array([cx_r, cy_r])).astype(np.int32)
            mask_r = np.zeros((dh, dw), dtype=np.uint8)
            cv2.fillPoly(mask_r, [pts_r], 255)
            m3_r = mask_r[:, :, np.newaxis] / 255.0
            overlay[:] = (overlay * (1 - m3_r * 0.65) + np.array([120, 80, 220]) * m3_r * 0.65).astype(np.uint8)
            cv2.polylines(overlay, [pts_r], True, (80, 40, 200), 2)

        elif s_orient == "Lancetta":
            cx = int(s.get("cx", 50) / 100 * dw)
            cy = int(s.get("cy", 50) / 100 * dh)
            angle_rad = np.deg2rad(s.get("angle", 90))
            length_px = int(s.get("length", 50) / 100 * max(dh, dw))
            ex = int(cx + np.cos(angle_rad) * length_px)
            ey = int(cy - np.sin(angle_rad) * length_px)
            thickness = max(2, int(s.get("size", 15) * dw / 1000))
            cv2.line(overlay, (cx, cy), (ex, ey), (120, 80, 220), thickness)
            cv2.circle(overlay, (cx, cy), max(3, thickness), (80, 40, 200), -1)

        elif s_orient == "Cerchio":
            cx = int(s.get("cx", 50) / 100 * dw)
            cy = int(s.get("cy", 50) / 100 * dh)
            radius = max(1, int(s.get("radius", 30) / 100 * max(dh, dw) / 2))
            if s.get("filled", True):
                mask_c = np.zeros((dh, dw), dtype=np.uint8)
                cv2.circle(mask_c, (cx, cy), radius, 255, -1)
                m3 = mask_c[:, :, np.newaxis] / 255.0
                overlay[:] = (overlay * (1 - m3 * 0.65) + VIOLET * m3 * 0.65).astype(np.uint8)
            else:
                cv2.circle(overlay, (cx, cy), radius, (120, 80, 220), max(2, s.get("size", 8)))
    return overlay


def draw_lancetta(out, src_stripe, h, w, cx_pct, cy_pct, angle_deg,
                  length_pct, thickness_px, chroma_on, chroma_amt, mode, opacity):
    """Disegna una striscia ruotata (lancetta) usando una maschera OpenCV."""
    cx = int(cx_pct / 100.0 * w)
    cy = int(cy_pct / 100.0 * h)
    length = int(length_pct / 100.0 * max(h, w))
    half_t = max(1, thickness_px // 2)

    angle_rad = np.deg2rad(angle_deg)
    ex = int(cx + np.cos(angle_rad) * length)
    ey = int(cy - np.sin(angle_rad) * length)  # y invertita (0 = cima)

    # Maschera della lancetta come rettangolo ruotato
    mask = np.zeros((h, w), dtype=np.uint8)
    dx = ex - cx
    dy = ey - cy
    norm = max(np.sqrt(dx*dx + dy*dy), 1e-6)
    px = int(-dy / norm * half_t)
    py = int( dx / norm * half_t)

    pts = np.array([
        [cx + px, cy + py],
        [cx - px, cy - py],
        [ex - px, ey - py],
        [ex + px, ey + py],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    patch = src_stripe.copy()
    if chroma_on and chroma_amt > 0:
        patch = apply_chroma(patch, chroma_amt)

    blended = blend_patch(out, patch, mode, opacity)
    mask3 = mask[:, :, np.newaxis] / 255.0
    out[:] = (out * (1.0 - mask3) + blended * mask3).astype(np.uint8)


def draw_cerchio(out, src_stripe, h, w, cx_pct, cy_pct, radius_pct,
                 filled, thickness_px, chroma_on, chroma_amt, mode, opacity):
    """Disegna un cerchio (pieno o bordo) come maschera."""
    cx = int(cx_pct / 100.0 * w)
    cy = int(cy_pct / 100.0 * h)
    radius = int(radius_pct / 100.0 * max(h, w) / 2.0)
    radius = max(1, radius)

    mask = np.zeros((h, w), dtype=np.uint8)
    if filled:
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    else:
        t = max(1, thickness_px)
        cv2.circle(mask, (cx, cy), radius, 255, t)

    patch = src_stripe.copy()
    if chroma_on and chroma_amt > 0:
        patch = apply_chroma(patch, chroma_amt)

    blended = blend_patch(out, patch, mode, opacity)
    mask3 = mask[:, :, np.newaxis] / 255.0
    out[:] = (out * (1.0 - mask3) + blended * mask3).astype(np.uint8)



def draw_striscia_ruotata(out, src_stripe, h, w, cx_pct, cy_pct, angle_deg,
                          spessore_pct, lunghezza_pct, chroma_on, chroma_amt, mode, opacity):
    """
    Striscia rettangolare larga che ruota attorno al suo centro.
    cx_pct, cy_pct: centro di rotazione (%)
    angle_deg: angolo 0-360 (0=orizzontale, 90=verticale, 45=diagonale)
    spessore_pct: altezza del rettangolo (% dell'immagine)
    lunghezza_pct: larghezza del rettangolo (% dell'immagine)
    """
    cx = int(cx_pct / 100.0 * w)
    cy = int(cy_pct / 100.0 * h)
    half_h = int(spessore_pct / 100.0 * max(h, w) / 2.0)
    half_w = int(lunghezza_pct / 100.0 * max(h, w) / 2.0)
    half_h = max(1, half_h)
    half_w = max(1, half_w)

    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # 4 angoli del rettangolo non ruotato, centrato in (cx, cy)
    corners = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h],
    ], dtype=np.float32)

    # Rotazione 2D
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = (rot_matrix @ corners.T).T
    pts = (rotated + np.array([cx, cy])).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    patch = src_stripe.copy()
    if chroma_on and chroma_amt > 0:
        patch = apply_chroma(patch, chroma_amt)

    blended = blend_patch(out, patch, mode, opacity)
    mask3 = mask[:, :, np.newaxis] / 255.0
    out[:] = (out * (1.0 - mask3) + blended * mask3).astype(np.uint8)


def resolve_reactive_opacity(s_dict, base_opacity, beat_gate_val, beat_sync_on, auto_key):
    """
    Con 'beat_react' attivo, beat_sync globale ON, e il moto automatico (auto_key)
    spento, la striscia lampeggia a tempo: piena visibilità (base_opacity) durante
    la finestra 'on' del beat_gate, invisibile (0) altrimenti — uno strobe netto.
    beat_gate è on/off puro derivato SOLO dal periodo del BPM (calcolato o manuale),
    non dal 'Beat Decay' del preset genere: la durata del lampo non dipende dal genere.
    """
    if beat_sync_on and s_dict.get('beat_react', False) and not s_dict.get(auto_key, False):
        return base_opacity if beat_gate_val > 0.5 else 0.0
    return base_opacity


def apply_stripe_window(bg_frame, calder_clean, calder_glitch, h, w,
                        stripes, stripe_orientation, stripe_glitch,
                        stripe_reverse=False, audio_envelope_val=1.0,
                        stripe_offsets=None,
                        stripe_chroma=False, stripe_flash=False,
                        beat_val=0.0, beat_gate_val=0.0, beat_sync_on=False,
                        t=0.0, total_dur=10.0, extra_sources=None):
    """
    stripes: lista di dict con keys: center, size, length, length_audio,
             move_random, move_speed, offset_length, chroma_amount,
             blend_mode, opacity, source
    stripe_chroma:  aberrazione cromatica dentro la striscia
    stripe_flash:   striscia si spegne (mostra bg) sui beat forti
    beat_val:       valore beat envelope corrente (0-1), dipende dal 'Beat Decay' del genere
    beat_gate_val:  on/off puro (0 o 1) legato solo al periodo del BPM, per lo strobe
    beat_sync_on:   True se 'A tempo di musica' è attivo con audio caricato
    extra_sources:  dict {nome_sorgente: frame_rgb} — Calderoni extra / Foto Fisse selezionabili
                    per singola striscia via s['source']. 'Calderone' (il comportamento di
                    sempre) è sempre disponibile e riflette stripe_reverse dinamicamente.
    """
    if stripe_offsets is None:
        stripe_offsets = [50.0] * len(stripes)

    # flash: se beat forte, la striscia scompare (mostra sfondo)
    flash_active = stripe_flash and beat_val > 0.7

    src_calder = calder_glitch if stripe_glitch else calder_clean

    if stripe_reverse:
        out = src_calder.copy()
        src_stripe = bg_frame
    else:
        out = bg_frame.copy()
        src_stripe = src_calder

    sources = {'Calderone': src_stripe}
    if extra_sources:
        sources.update(extra_sources)

    def _paste_h(src, p0, p1, l0, l1, chroma_amt, mode, opacity):
        if p1 > p0 and l1 > l0:
            patch = src[p0:p1, l0:l1].copy()
            if stripe_chroma and chroma_amt > 0:
                patch = apply_chroma(patch, chroma_amt)
            base_patch = out[p0:p1, l0:l1]
            out[p0:p1, l0:l1] = blend_patch(base_patch, patch, mode, opacity)

    def _paste_v(src, p0, p1, l0, l1, chroma_amt, mode, opacity):
        if p1 > p0 and l1 > l0:
            patch = src[l0:l1, p0:p1].copy()
            if stripe_chroma and chroma_amt > 0:
                patch = apply_chroma(patch, chroma_amt)
            base_patch = out[l0:l1, p0:p1]
            out[l0:l1, p0:p1] = blend_patch(base_patch, patch, mode, opacity)

    def _draw(s, offset, is_h, src):
        if flash_active:
            return  # striscia spenta sul beat
        center   = kf_get(s, 'center', t, total_dur, s.get('center', 50))
        size     = kf_get(s, 'size', t, total_dur, s.get('size', 10))
        base_len = kf_get(s, 'length', t, total_dur, s.get('length', 100))
        if s.get('length_audio', False):
            base_len = base_len * (0.2 + 0.8 * audio_envelope_val)
        base_len = np.clip(base_len, 1.0, 100.0)
        length_offset = resolve_reactive_val(s, s.get('offset_length', 50.0), offset, 'move_random')
        chroma_amt = int(s.get('chroma_amount', 6))
        blend_mode = s.get('blend_mode', 'Normal')
        opacity    = kf_get(s, 'opacity', t, total_dur, float(s.get('opacity', 1.0)))
        opacity    = resolve_reactive_opacity(s, opacity, beat_gate_val, beat_sync_on, 'move_random')
        dim = (h, w) if is_h else (w, h)
        p0, p1, l0, l1 = compute_stripe_coords(center, size, base_len, length_offset, dim)
        if is_h:
            _paste_h(src, p0, p1, l0, l1, chroma_amt, blend_mode, opacity)
        else:
            _paste_v(src, p0, p1, l0, l1, chroma_amt, blend_mode, opacity)

    for idx, s in enumerate(stripes):
        if flash_active:
            continue
        offset = stripe_offsets[idx]
        s_orient = s.get('orientation', stripe_orientation)
        chroma_amt = int(s.get('chroma_amount', 6))
        mode    = s.get('blend_mode', 'Normal')
        opacity = float(s.get('opacity', 1.0))
        chroma_on = stripe_chroma and chroma_amt > 0
        this_src = sources.get(s.get('source', 'Calderone'), src_stripe)

        if s_orient == "Lancetta":
            # angolo base + rotazione automatica nel tempo (offset usato come angolo corrente)
            angle_base = kf_get(s, 'angle', t, total_dur, s.get('angle', 90.0))
            angle = resolve_reactive_val(s, angle_base, offset, 'auto_rotate')
            length_pct = kf_get(s, 'length', t, total_dur, s.get('length', 50.0))
            if s.get('length_audio', False):
                length_pct = length_pct * (0.2 + 0.8 * audio_envelope_val)
            opacity = kf_get(s, 'opacity', t, total_dur, float(s.get('opacity', 1.0)))
            opacity = resolve_reactive_opacity(s, opacity, beat_gate_val, beat_sync_on, 'auto_rotate')
            draw_lancetta(out, this_src, h, w,
                          s.get('cx', 50.0), s.get('cy', 50.0),
                          angle, length_pct,
                          int(kf_get(s, 'size', t, total_dur, s.get('size', 10))),
                          chroma_on, chroma_amt, mode, opacity)

        elif s_orient == "Cerchio":
            radius = kf_get(s, 'radius', t, total_dur, s.get('radius', 30.0))
            if s.get('length_audio', False):
                radius = radius * (0.2 + 0.8 * audio_envelope_val)
            radius = resolve_reactive_val(s, radius, offset, 'auto_expand')
            opacity = kf_get(s, 'opacity', t, total_dur, float(s.get('opacity', 1.0)))
            opacity = resolve_reactive_opacity(s, opacity, beat_gate_val, beat_sync_on, 'auto_expand')
            draw_cerchio(out, this_src, h, w,
                         s.get('cx', 50.0), s.get('cy', 50.0),
                         radius, s.get('filled', True),
                         int(kf_get(s, 'size', t, total_dur, s.get('size', 8))),
                         chroma_on, chroma_amt, mode, opacity)

        elif s_orient == "Striscia Ruotata":
            angle_base = kf_get(s, 'angle', t, total_dur, s.get('angle', 0.0))
            angle = resolve_reactive_val(s, angle_base, offset, 'auto_rotate')
            length_pct = kf_get(s, 'length', t, total_dur, s.get('length', 100.0))
            if s.get('length_audio', False):
                length_pct = length_pct * (0.2 + 0.8 * audio_envelope_val)
            opacity = kf_get(s, 'opacity', t, total_dur, float(s.get('opacity', 1.0)))
            opacity = resolve_reactive_opacity(s, opacity, beat_gate_val, beat_sync_on, 'auto_rotate')
            draw_striscia_ruotata(out, this_src, h, w,
                                  s.get('cx', 50.0), s.get('cy', 50.0),
                                  angle,
                                  float(kf_get(s, 'size', t, total_dur, s.get('size', 15.0))),
                                  length_pct,
                                  chroma_on, chroma_amt, mode, opacity)

        elif s_orient in ("Orizzontale", "Verticale"):
            _draw(s, offset, s_orient == "Orizzontale", this_src)

    return out


def get_or_decode_audio(up_aud, duration, sr_target=22050):
    """
    Decodifica l'audio in cache (session_state), chiave = file + durata richiesta.
    Evita di ridecodificare l'mp3 due volte (anteprima BPM + generazione finale)
    quando file e durata non sono cambiati.
    """
    key = f"{up_aud.name}_{up_aud.size}_{round(duration, 2)}"
    cache = st.session_state.get("audio_decode_cache")
    if cache and cache.get("key") == key:
        return cache["y"], cache["sr"]

    up_aud.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as t:
        t.write(up_aud.read())
        path = t.name
    up_aud.seek(0)  # riporto il puntatore a inizio file per eventuali letture successive
    y, sr = librosa.load(path, sr=sr_target, mono=True, duration=duration)
    os.remove(path)

    st.session_state["audio_decode_cache"] = {"key": key, "y": y, "sr": sr}
    return y, sr


def analyze_audio(y, sr, total_f, fps, beat_sync, slideshow_mode, genre, manual_bpm, onset_sensitivity):
    """
    Unica fonte di verità per l'analisi audio: usata sia dall'anteprima BPM/grafico in UI
    sia da generate_master, così i due punti non possono andare fuori sincrono tra loro.
    Ritorna un dict con inviluppi e metadati derivati dall'audio.
    """
    result = {
        'audio_envelope': np.ones(total_f),
        'beat_envelope': np.zeros(total_f),
        'beat_gate': np.zeros(total_f),
        'beat_phase': np.zeros(total_f),
        'onset_envelope': np.zeros(total_f),
        'rhythm_envelope': None,
        'audio_peak': 0.0,
        'beat_times': np.array([]),
        'onset_times': np.array([]),
        'detected_bpm': 0.0,
        'bpm_source': "N/A",
        'bs': 0, 'bd': 50, 'op': 0, 'bc': 30,
        'rhythm_tracking': False,
    }

    rms = librosa.feature.rms(y=y)[0]
    result['audio_peak'] = float(np.max(rms))
    result['audio_envelope'] = np.interp(
        np.linspace(0, len(rms) - 1, total_f),
        np.arange(len(rms)), rms / (rms.max() + 1e-6)
    )

    if beat_sync and not slideshow_mode:
        p = GENRE_PRESETS[genre]
        bs, bd, op, bc = p["beat_strength"], p["beat_decay"], p["onset"], p["cache"]
        rhythm_tracking = p["rhythm"]
        result.update(bs=bs, bd=bd, op=op, bc=bc, rhythm_tracking=rhythm_tracking)

        if bs > 0 or manual_bpm:
            if manual_bpm:
                # --- GRIGLIA BPM MANUALE ---
                detected_bpm = float(manual_bpm)
                bpm_source = "MANUALE"
                step = 60.0 / detected_bpm
                beat_times = np.arange(0, len(y) / sr, step)
            else:
                # --- BPM AUTO-DETECT ---
                onset_env_tempo = librosa.onset.onset_strength(y=y, sr=sr)
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env_tempo)
                detected_bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
                bpm_source = "AUTO"
                beat_times = librosa.frames_to_time(beat_frames, sr=sr)

            result['beat_times']   = beat_times
            result['detected_bpm'] = detected_bpm
            result['bpm_source']   = bpm_source

            # Se il primo beat rilevato non è vicino a t=0 (tipico dell'auto-detect, che spesso
            # aggancia il tempo solo dopo una frazione di secondo), estendo la griglia all'indietro
            # fino all'inizio del brano assumendo lo stesso tempo — altrimenti l'effetto a beat
            # (opacità, movimento) resta fermo/spento per i primi frame invece di partire subito.
            if len(beat_times) >= 1 and beat_times[0] > 1e-6:
                period_start = (float(np.median(np.diff(beat_times))) if len(beat_times) >= 2
                                else 60.0 / detected_bpm if detected_bpm > 0 else None)
                if period_start and period_start > 1e-6:
                    n_prepend = int(np.floor(beat_times[0] / period_start))
                    if n_prepend > 0:
                        prepend_times = beat_times[0] - period_start * np.arange(n_prepend, 0, -1)
                        prepend_times = prepend_times[prepend_times >= 0]
                        beat_times = np.concatenate([prepend_times, beat_times])
                        result['beat_times'] = beat_times

            decay_rate = 1.0 - (bd / 100.0) * 0.98
            for bt in beat_times:
                bf = int(bt * fps)
                for df in range(min(int(fps * 0.5), total_f - bf)):
                    result['beat_envelope'][bf + df] = max(result['beat_envelope'][bf + df], decay_rate ** df)

            # beat_gate: on/off netto per lo strobe, legato SOLO al periodo del BPM
            # (non al 'Beat Decay' del preset genere, che invece modella beat_envelope sopra)
            period = 60.0 / detected_bpm if detected_bpm > 0 else 0.5
            on_frames = max(1, int(round(period * 0.25 * fps)))
            for bt in beat_times:
                bf = int(bt * fps)
                for df in range(min(on_frames, total_f - bf)):
                    result['beat_gate'][bf + df] = 1.0

            # beat_phase: quanti beat sono trascorsi, in modo continuo (non a scatti).
            # Usata come "orologio musicale" per il movimento delle strisce sincronizzate:
            # a 140 BPM avanza esattamente il doppio più veloce che a 70 BPM, sempre.
            if len(beat_times) >= 2:
                period_med = float(np.median(np.diff(beat_times)))
                ext_times = np.concatenate(([beat_times[0] - period_med], beat_times, [beat_times[-1] + period_med]))
                ext_idx = np.arange(-1, len(beat_times) + 1, dtype=float)
                frame_times = np.arange(total_f) / fps
                result['beat_phase'] = np.interp(frame_times, ext_times, ext_idx)
            elif len(beat_times) == 1 and detected_bpm > 0:
                frame_times = np.arange(total_f) / fps
                result['beat_phase'] = (frame_times - beat_times[0]) / (60.0 / detected_bpm)

        # --- ONSET DETECTION (ogni transiente, indipendente dal tempo) ---
        sensitivity = onset_sensitivity if onset_sensitivity is not None else (op / 100.0)
        if sensitivity > 0:
            onset_env_raw = librosa.onset.onset_strength(y=y, sr=sr)
            onset_env_norm = onset_env_raw / (onset_env_raw.max() + 1e-6)
            # delta più basso = più sensibile (intercetta più transienti, anche deboli)
            delta = 0.6 * (1.0 - sensitivity) + 0.02
            # frame del PICCO (per leggere l'intensità reale del transiente)
            onset_frames_peak = librosa.onset.onset_detect(
                onset_envelope=onset_env_norm, sr=sr, backtrack=False, delta=delta
            )
            # frame dell'ATTACCO reale (per il timing preciso del taglio)
            onset_frames_bt = librosa.onset.onset_detect(
                onset_envelope=onset_env_norm, sr=sr, backtrack=True, delta=delta
            )
            onset_times = librosa.frames_to_time(onset_frames_bt, sr=sr)
            result['onset_times'] = onset_times
            for peak_frame, ot in zip(onset_frames_peak, onset_times):
                of = int(ot * fps)
                if of < total_f:
                    strength = float(onset_env_norm[peak_frame])
                    result['onset_envelope'][of] = max(result['onset_envelope'][of], strength)

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
                np.linspace(0, len(combined) - 1, total_f),
                np.arange(len(combined)), combined
            )
            kernel = np.ones(fps // 2) / (fps // 2)
            rhythm_envelope = np.convolve(rhythm_envelope, kernel, mode='same')
            rhythm_envelope = np.clip(rhythm_envelope / (rhythm_envelope.max() + 1e-6), 0.0, 1.0)
            result['rhythm_envelope'] = rhythm_envelope

    return result


def generate_master(up_m1, up_m2, up_trit, up_aud,
                    orientation, strand_val, max_limit,
                    chaos_val, photo_speed, format_type,
                    m1_end, m2_start,
                    rand_lines,
                    beat_sync, genre,
                    seq_mode,
                    slideshow_mode, slide_hold, slide_trans, slide_trans_type,
                    stripe_mode=False, stripes=None, stripe_orientation="Orizzontale",
                    stripe_bg="Master 1", stripe_glitch=False, stripe_reverse=False,
                    stripe_chroma=False, stripe_flash=False,
                    global_chroma=False, global_chroma_amt=6,
                    global_flash=False, global_flash_threshold=0.7, global_flash_intensity=100,
                    manual_bpm=None, onset_sensitivity=None,
                    calderone2_cfg=None,
                    bg_source="Calderone (originale)", bg_static_file=None, bg_video_file=None,
                    overlays_cfg=None):

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

    # --- CALDERONE 2 (opzionale): stesso trattamento glitch del Calderone 1, si mescolano
    # gradualmente in base al progresso — stesso meccanismo di dissolvenza di Master 1/2.
    pool_imgs2 = []
    calderone2_mix_from = 0.5
    if calderone2_cfg:
        _c2_files = calderone2_cfg.get('files') or []
        if _c2_files:
            pool_imgs2 = [load_img_half(f) for f in _c2_files]
            calderone2_mix_from = calderone2_cfg.get('mix_from', 0.5)

    def pick_calderone_pool(prog):
        """Se il Calderone 2 è configurato, sfuma gradualmente dal pool 1 al pool 2
        dopo 'calderone2_mix_from' (stesso meccanismo di dissolvenza di Master1/2).
        Altrimenti resta sempre sul pool 1 — comportamento originale invariato."""
        if not pool_imgs2:
            return pool_imgs
        if prog < calderone2_mix_from:
            return pool_imgs
        span = max(1.0 - calderone2_mix_from, 1e-6)
        ramp = np.clip((prog - calderone2_mix_from) / span, 0.0, 1.0)
        return pool_imgs2 if random.random() < ramp else pool_imgs

    # --- SFONDO CUSTOM: Foto Fissa o Video, sostituisce Calderone/Master per tutta la durata ---
    bg_static_frame = None
    bg_video_cap = None
    if bg_source == "Foto Fissa" and bg_static_file:
        bg_static_frame = load_img_half(bg_static_file)
    elif bg_source == "Video" and bg_video_file:
        bg_video_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(bg_video_file.name)[1]) as _tmpbg:
            _tmpbg.write(bg_video_file.read())
            _bg_tmp_path = _tmpbg.name
        bg_video_cap = cv2.VideoCapture(_bg_tmp_path)

    def get_custom_bg_frame(cur_t):
        """Foto Fissa: sempre la stessa immagine. Video: fotogramma al tempo cur_t (loop se
        più corto della durata). Nessun effetto/glitch — è lo sfondo così com'è."""
        if bg_source == "Foto Fissa" and bg_static_frame is not None:
            return bg_static_frame
        if bg_source == "Video" and bg_video_cap is not None:
            frame_rgb = get_video_overlay_frame(bg_video_cap, cur_t)
            if frame_rgb is not None:
                if frame_rgb.shape[:2] != (h, w):
                    frame_rgb = cv2.resize(frame_rgb, (w, h))
                return frame_rgb
        return pool_imgs[0]  # fallback di sicurezza se il file non è valido

    # --- OVERLAY FOTO/VIDEO: solo posizione e dimensione, nessun effetto/blend/pulsazione ---
    overlays_prepared = []  # {kind:'image'|'video', rgba (image) o cap (video), cx, cy, scale}
    if overlays_cfg:
        for _ov in overlays_cfg:
            _ovf = _ov.get('file')
            if _ovf is None:
                continue
            _name = _ovf.name.lower()
            if _name.endswith(('.mp4', '.mov', '.avi', '.mkv')):
                _ovf.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(_name)[1]) as _tmpv:
                    _tmpv.write(_ovf.read())
                    _tmp_path = _tmpv.name
                _cap = cv2.VideoCapture(_tmp_path)
                overlays_prepared.append({'kind': 'video', 'cap': _cap,
                                           'cx': _ov['cx'], 'cy': _ov['cy'], 'scale': _ov['scale']})
            else:
                overlays_prepared.append({'kind': 'image', 'rgba': load_overlay_image(_ovf),
                                           'cx': _ov['cx'], 'cy': _ov['cy'], 'scale': _ov['scale']})

    def build_overlay_frame_list(cur_t):
        """Per ogni overlay ottiene l'rgba corrente: statico per le foto, estratto dal
        video (con loop) per i video — nessun effetto, solo il fotogramma così com'è."""
        out_list = []
        for ovp in overlays_prepared:
            if ovp['kind'] == 'video':
                frame_rgb = get_video_overlay_frame(ovp['cap'], cur_t)
                if frame_rgb is None:
                    continue
                alpha_full = np.full(frame_rgb.shape[:2] + (1,), 255, dtype=np.uint8)
                rgba = np.dstack([frame_rgb, alpha_full])
            else:
                rgba = ovp['rgba']
            out_list.append({'rgba': rgba, 'scale': ovp['scale'], 'cx': ovp['cx'], 'cy': ovp['cy']})
        return out_list

    img_m1_half = load_img_half(up_m1) if up_m1 else None
    img_m2_half = load_img_half(up_m2) if up_m2 else None

    if format_type == "16:9 (Orizzontale)": out_w, out_h = 1280, 720
    elif format_type == "9:16 (Verticale)":  out_w, out_h = 720, 1280
    else:                                     out_w, out_h = 1080, 1080

    # --- sfondo per stripe mode ---
    stripe_bg_static  = None   # Master 1/2 fisso
    stripe_use_render = (stripe_bg == "Render")  # usa frame glitchato come sfondo

    if stripe_mode and stripes and not stripe_use_render:
        if stripe_bg == "Master 1" and up_m1 is not None:
            up_m1.seek(0)
            _bg_raw = resize_to_format(np.array(Image.open(up_m1).convert("RGB")), format_type, half_res=True)
            stripe_bg_static = cv2.resize(_bg_raw, (w, h))
        elif stripe_bg == "Master 2" and up_m2 is not None:
            up_m2.seek(0)
            _bg_raw = resize_to_format(np.array(Image.open(up_m2).convert("RGB")), format_type, half_res=True)
            stripe_bg_static = cv2.resize(_bg_raw, (w, h))
        # "Calderone" → stripe_bg_static rimane None, si usa pick() a runtime

    # --- traiettorie strisce: calcolate dopo l'analisi audio (vedi sotto), così possono reagire al beat ---
    stripe_offsets_t = []

    # --- AUDIO ANALYSIS ---
    audio_envelope  = np.ones(total_f)
    beat_envelope   = np.zeros(total_f)
    beat_gate       = np.zeros(total_f)
    beat_phase      = np.zeros(total_f)
    onset_envelope  = np.zeros(total_f)
    rhythm_envelope = None
    audio_peak      = 0.0
    bs, bc          = 0, 30
    detected_bpm    = 0.0
    bpm_source      = "N/A"
    temp_aud_path   = None

    if up_aud:
        y, sr = get_or_decode_audio(up_aud, max_limit)
        audio_result = analyze_audio(
            y, sr, total_f, fps, beat_sync, slideshow_mode, genre, manual_bpm, onset_sensitivity
        )
        audio_envelope  = audio_result['audio_envelope']
        beat_envelope   = audio_result['beat_envelope']
        beat_gate       = audio_result['beat_gate']
        beat_phase      = audio_result['beat_phase']
        onset_envelope  = audio_result['onset_envelope']
        rhythm_envelope = audio_result['rhythm_envelope']
        audio_peak      = audio_result['audio_peak']
        bs, bc          = audio_result['bs'], audio_result['bc']
        detected_bpm    = audio_result['detected_bpm']
        bpm_source      = audio_result['bpm_source']

        # file temporaneo persistente per il muxing audio finale (AudioFileClip più sotto)
        up_aud.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as t:
            t.write(up_aud.read())
            temp_aud_path = t.name
        up_aud.seek(0)

    # --- traiettorie strisce (ora che beat_envelope/onset_envelope esistono) ---
    if stripe_mode and stripes:
        rng = np.random.default_rng(42)
        t_arr = np.linspace(0, max_limit, total_f)
        # orologio musicale: quanti beat sono trascorsi, in modo continuo.
        # A 140 BPM avanza esattamente il doppio più veloce che a 70 BPM — non un'accelerazione
        # approssimativa, ma un aggancio diretto e proporzionale al tempo reale del brano.
        beat_clock = beat_phase

        def _ramp_or_pulse(s, cfg, base_t):
            """Rampa continua se il moto automatico (auto_key) è ON; altrimenti resta ferma
            al valore base. Se la striscia è anche 'Sincronizza al beat', base_t è l'orologio
            musicale (beat_clock): la velocità impostata (spd) si intende allora 'per beat',
            quindi raddoppiando il BPM il movimento raddoppia di velocità automaticamente."""
            base_val = s.get(cfg["base_key"], cfg["base_default"])
            if s.get(cfg["auto_key"], False):
                spd = s.get(cfg["speed_key"], cfg["speed_default"])
                ramp = spd * base_t
                if cfg["add_base_to_ramp"]:
                    ramp = base_val + ramp
                return ramp % cfg["wrap_hi"]
            return np.full(total_f, base_val)

        for s in stripes:
            s_orient = s.get('orientation', stripe_orientation)
            react = bool(s.get('beat_react', False)) and beat_sync and not slideshow_mode
            base_t = beat_clock if react else t_arr

            if s_orient in STRIPE_MOTION_CONFIG:
                traj = _ramp_or_pulse(s, STRIPE_MOTION_CONFIG[s_orient], base_t)
                stripe_offsets_t.append(traj)

            elif s_orient in ("Orizzontale", "Verticale"):
                base_offset = s.get('offset_length', 50.0)
                if s.get('move_random', False):
                    spd = max(0.1, s.get('move_speed', 1.0))
                    freq1 = spd * rng.uniform(0.1, 0.3)
                    freq2 = spd * rng.uniform(0.05, 0.15)
                    phase1, phase2 = rng.uniform(0, np.pi*2, 2)
                    traj = (np.sin(2*np.pi*freq1*base_t + phase1) * 0.5 +
                            np.sin(2*np.pi*freq2*base_t + phase2) * 0.5)
                    traj = (traj + 1) / 2 * 80 + 10
                else:
                    # ferma al suo posto: senza moto automatico, "Sincronizza al beat" non muove nulla qui
                    traj = np.full(total_f, base_offset)
                stripe_offsets_t.append(traj)

            else:
                stripe_offsets_t.append(np.full(total_f, 50.0))


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

    # --- finalizzazione frame: overlay foto/video sopra lo sfondo, poi resize export ---
    def _finalize(raw_frame, f, t):
        if overlays_prepared:
            # canvas = dimensioni REALI del frame corrente, non h/w fissi: img_m1/img_m2
            # sono a piena risoluzione (out_w x out_h), mentre il resto della pipeline
            # (pick(), stripe, glitch) lavora a mezza risoluzione (h x w) — usare h/w fissi
            # qui avrebbe causato un mismatch di shape (crash) sui frame Master 1/2.
            ch, cw = raw_frame.shape[:2]
            ov_list = build_overlay_frame_list(t)
            if ov_list:
                raw_frame = apply_media_overlays(raw_frame, ov_list, ch, cw)
        return cv2.resize(raw_frame, (out_w, out_h))

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

            _aenv = float(audio_envelope[f])
            _soff = [stripe_offsets_t[si][f] for si in range(len(stripes))] if stripes else []
            _bval = float(beat_envelope[f])
            _bgate = float(beat_gate[f])

            def _get_bg_slide(glitched_frame=None):
                if stripe_use_render and glitched_frame is not None:
                    return glitched_frame
                return stripe_bg_static if stripe_bg_static is not None else img_next

            if cycle_pos < slide_hold:
                if stripe_mode and stripes:
                    out_frame = apply_stripe_window(_get_bg_slide(img_cur), img_cur, img_cur, h, w,
                                                    stripes, stripe_orientation, False,
                                                    stripe_reverse, _aenv, _soff,
                                                    stripe_chroma, stripe_flash, _bval, _bgate,
                                                    t=t, total_dur=max_limit, beat_sync_on=(beat_sync and up_aud is not None))
                else:
                    out_frame = img_cur
                return _finalize(out_frame, f, t)
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
                    if stripe_mode and stripes:
                        out_frame = apply_stripe_window(_get_bg_slide(glitched), base, glitched, h, w,
                                                        stripes, stripe_orientation, stripe_glitch,
                                                        stripe_reverse, _aenv, _soff,
                                                        stripe_chroma, stripe_flash, _bval, _bgate,
                                                        t=t, total_dur=max_limit, beat_sync_on=(beat_sync and up_aud is not None))
                    else:
                        out_frame = glitched
                else:
                    intensity = np.sin(trans_prog * np.pi)
                    blend = (img_cur * (1.0 - trans_prog) + img_next * trans_prog).astype(np.uint8)
                    glitched = apply_glitch_stripes(blend, blend, h, w, orientation, strand_val, rand_lines, intensity)
                    if stripe_mode and stripes:
                        out_frame = apply_stripe_window(_get_bg_slide(glitched), blend, glitched, h, w,
                                                        stripes, stripe_orientation, stripe_glitch,
                                                        stripe_reverse, _aenv, _soff,
                                                        stripe_chroma, stripe_flash, _bval, _bgate,
                                                        t=t, total_dur=max_limit, beat_sync_on=(beat_sync and up_aud is not None))
                    else:
                        out_frame = glitched

                return _finalize(out_frame, f, t)

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

            if bg_source != "Calderone (originale)":
                _bg_content = get_custom_bg_frame(t)
                if stripe_mode and stripes:
                    _aenv  = float(audio_envelope[f])
                    _bval  = float(beat_envelope[f])
                    _bgate = float(beat_gate[f])
                    _soff  = [stripe_offsets_t[si][f] for si in range(len(stripes))] if stripes else []
                    out_frame = apply_stripe_window(_bg_content, _bg_content, _bg_content, h, w,
                                                    stripes, stripe_orientation, False, stripe_reverse,
                                                    _aenv, _soff, stripe_chroma, stripe_flash, _bval, _bgate,
                                                    t=t, total_dur=max_limit,
                                                    beat_sync_on=(beat_sync and up_aud is not None))
                else:
                    out_frame = _bg_content
                return _finalize(out_frame, f, t)

            has_masters = (img_m1 is not None) and (img_m2 is not None)

            if has_masters:
                if prog <= m1_end:
                    _ramp_m1 = np.clip(prog / m1_end if m1_end > 0.001 else 1.0, 0.0, 1.0)
                    if _ramp_m1 < 0.02:
                        return _finalize(img_m1, f, t)
                    _m1_prob = 1.0 - _ramp_m1
                    def pick():
                        key = f // max(1, int(fps / photo_speed))
                        if key in cached_picks and random.random() > 0.1:
                            return cached_picks[key]
                        if random.random() < _m1_prob:
                            res = img_m1_half
                        else:
                            _active_pool = pick_calderone_pool(prog)
                            idx = (key % len(_active_pool)) if seq_mode else None
                            res = _active_pool[idx] if seq_mode else random.choice(_active_pool)
                        cache_set(key, res)
                        return res
                elif prog >= m2_start:
                    _span_m2 = 1.0 - m2_start if m2_start < 0.999 else 1e-6
                    _ramp_m2 = np.clip((prog - m2_start) / _span_m2, 0.0, 1.0)
                    if _ramp_m2 > 0.98:
                        return _finalize(img_m2, f, t)
                    _m2_prob = _ramp_m2
                    def pick():
                        key = f // max(1, int(fps / photo_speed))
                        if key in cached_picks and random.random() > 0.1:
                            return cached_picks[key]
                        if random.random() < _m2_prob:
                            res = img_m2_half
                        else:
                            _active_pool = pick_calderone_pool(prog)
                            idx = (key % len(_active_pool)) if seq_mode else None
                            res = _active_pool[idx] if seq_mode else random.choice(_active_pool)
                        cache_set(key, res)
                        return res
                else:
                    def pick():
                        interval = max(1, int(fps / photo_speed))
                        key = f // interval
                        force = beat_sync and onset_envelope[f] > 0 and random.random() < (bc / 100.0) * onset_envelope[f]
                        if key in cached_picks and not force and random.random() > 0.1:
                            return cached_picks[key]
                        _active_pool = pick_calderone_pool(prog)
                        idx = (key % len(_active_pool)) if seq_mode else None
                        res = _active_pool[idx] if seq_mode else random.choice(_active_pool)
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
                    force = beat_sync and onset_envelope[f] > 0 and random.random() < (bc / 100.0) * onset_envelope[f]
                    if key in cached_picks and not force and random.random() > 0.1:
                        return cached_picks[key]
                    _active_pool = pick_calderone_pool(prog)
                    idx = (key % len(_active_pool)) if seq_mode else None
                    res = _active_pool[idx] if seq_mode else random.choice(_active_pool)
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

            _aenv = float(audio_envelope[f])
            _bval = float(beat_envelope[f])
            _bgate = float(beat_gate[f])
            _soff = [stripe_offsets_t[si][f] for si in range(len(stripes))] if stripes else []

            if orientation == "Nessun Effetto":
                if stripe_mode and stripes:
                    calder_clean = pick()
                    if stripe_use_render:
                        # nessun glitch → frame pulito come sfondo
                        _bg = calder_clean
                    else:
                        _bg = stripe_bg_static if stripe_bg_static is not None else pick()
                    return _finalize(
                        apply_stripe_window(_bg, calder_clean, calder_clean, h, w,
                                            stripes, stripe_orientation, False, stripe_reverse,
                                            _aenv, _soff, stripe_chroma, stripe_flash, _bval, _bgate,
                                            t=t, total_dur=max_limit, beat_sync_on=(beat_sync and up_aud is not None)),
                        f, t)
                return _finalize(pick(), f, t)

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
            if stripe_mode and stripes:
                if stripe_use_render:
                    _bg = frame.copy()
                else:
                    _bg = stripe_bg_static if stripe_bg_static is not None else pick()
                frame = apply_stripe_window(_bg, calder_clean, frame, h, w,
                                            stripes, stripe_orientation, stripe_glitch,
                                            stripe_reverse, _aenv, _soff,
                                            stripe_chroma, stripe_flash, _bval, _bgate,
                                            t=t, total_dur=max_limit, beat_sync_on=(beat_sync and up_aud is not None))

            # --- Effetti globali standalone (senza strisce selettive) ---
            if not stripe_mode:
                if global_chroma:
                    frame = apply_chroma(frame, global_chroma_amt)
                if global_flash and _bval > global_flash_threshold:
                    alpha = global_flash_intensity / 100.0
                    frame = (calder_clean * (1.0 - alpha) + frame * (1.0 - alpha)).astype(np.uint8)

            result = _finalize(frame, f, t)
            if f == 0 and 'preview_frame' not in st.session_state:
                st.session_state.preview_frame = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            return result

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

    slide_info_it = ""
    slide_info_en = ""
    if slideshow_mode:
        slide_info_it = f"""
* MODALITÀ: SLIDESHOW LENTO
* Durata foto: {slide_hold}s | Transizione: {slide_trans}s
* Tipo transizione: {slide_trans_type}
* Sequenza: {'ORDINATA' if seq_mode else 'RANDOM'}"""
        slide_info_en = f"""
* MODE: SLOW SLIDESHOW
* Photo duration: {slide_hold}s | Transition: {slide_trans}s
* Transition type: {slide_trans_type}
* Sequence: {'ORDERED' if seq_mode else 'RANDOM'}"""

    stripe_info_it = ""
    stripe_info_en = ""
    if stripe_mode and stripes:
        bg_label = "Frame" if stripe_bg == "Calderone" else stripe_bg
        stripe_info_it = f"""
* STRISCE SELETTIVE: {len(stripes)} striscia/e ({stripe_orientation})
* Sfondo: {bg_label} | Striscia: {'GLITCHATA' if stripe_glitch else 'ORIGINALE'}"""
        stripe_info_en = f"""
* SELECTIVE STRIPES: {len(stripes)} stripe(s) ({stripe_orientation})
* Background: {bg_label} | Stripe: {'GLITCHED' if stripe_glitch else 'ORIGINAL'}"""

    bpm_source_en = {"MANUALE": "MANUAL", "AUTO": "AUTO", "N/A": "N/A"}.get(bpm_source, bpm_source)

    report_text = f"""[SLICE_PHOTO_DISSECTION] // VOL_01 // H.264 // DATA_FRAGMENT

:: IT ::
:: MOTORE: recursive_cut_pro [v10.0 — keyframe]
:: EFFETTO: Recursive Strand Shift
:: ANALISI: RMS / Beat Sync / Rhythm Tracking
:: PROCESSO: Frammentazione Ricorsiva

"L'immagine e' stata smontata. Il codice ne ha riscritto la struttura."

TECHNICAL LOG SHEET:
* File: {base_name}
* Asset Pool: {len(pool_imgs)} foto
* Rendering: {total_f} frame @ {fps}fps
* Geometria: {orientation} @ {strand_val}px
* Chaos: {chaos_val}% | Photo Speed: {photo_speed}fps
* M1 sparisce a: {int(m1_end*100)}% | M2 appare a: {int(m2_start*100)}%
* Audio Peak: {audio_peak:.4f}
* Beat Sync: {'ON' if beat_sync and not slideshow_mode else 'OFF (Slideshow)' if slideshow_mode else 'OFF'}
* BPM: {f'{detected_bpm:.1f} ({bpm_source})' if beat_sync and not slideshow_mode and detected_bpm > 0 else 'N/A'}
* Onset Sensitivity: {f'{int(onset_sensitivity*100)}%' if beat_sync and not slideshow_mode and onset_sensitivity is not None else 'N/A (preset)'}
* Power Curve: {'BYPASSED' if rhythm_on else 'ON'}
* Sequenza Frame: {'ORDINATA' if seq_mode else 'RANDOM'}{slide_info_it}{stripe_info_it}

Regia e Algoritmo: Loop507

:: EN ::
[SLICE_PHOTO_DISSECTION] // VOL_01 // H.264 // DATA_FRAGMENT
:: ENGINE: recursive_cut_pro [v10.0 — keyframe]
:: EFFECT: Recursive Strand Shift
:: ANALYSIS: RMS / Beat Sync / Rhythm Tracking
:: PROCESS: Recursive Fragmentation

"The image has been disassembled. The code rewrote its structure."

TECHNICAL LOG SHEET:
* File: {base_name}
* Asset Pool: {len(pool_imgs)} photos
* Rendering: {total_f} frames @ {fps}fps
* Geometry: {orientation} @ {strand_val}px
* Chaos: {chaos_val}% | Photo Speed: {photo_speed}fps
* M1 fades out at: {int(m1_end*100)}% | M2 appears at: {int(m2_start*100)}%
* Audio Peak: {audio_peak:.4f}
* Beat Sync: {'ON' if beat_sync and not slideshow_mode else 'OFF (Slideshow)' if slideshow_mode else 'OFF'}
* BPM: {f'{detected_bpm:.1f} ({bpm_source_en})' if beat_sync and not slideshow_mode and detected_bpm > 0 else 'N/A'}
* Onset Sensitivity: {f'{int(onset_sensitivity*100)}%' if beat_sync and not slideshow_mode and onset_sensitivity is not None else 'N/A (preset)'}
* Power Curve: {'BYPASSED' if rhythm_on else 'ON'}
* Frame Sequence: {'ORDERED' if seq_mode else 'RANDOM'}{slide_info_en}{stripe_info_en}

Direction & Algorithm: Loop507

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

# dur/formato si scelgono in c3 ma servono in c2 (KF UI, anteprima livelli) — leggo
# direttamente la key del widget (con .get/fallback, MAI un'assegnazione manuale: Streamlit
# solleva un errore se una key di un widget con valore di default viene pre-impostata a mano).
# Streamlit aggiorna la key PRIMA di rieseguire lo script, quindi il valore letto qui è
# già quello corrente scelto dall'utente, non un passo indietro.

c1, c2, c3 = st.columns([1, 1.2, 1])
bottom_container = st.container()  # riga sotto: Strisce | Livelli affiancate
with c1:
    st.subheader("🖼️ Assets")

    bg_source = st.selectbox("🎨 Sfondo", ["Calderone (originale)", "Foto Fissa", "Video"],
        key="bg_source",
        help="Cosa genera lo sfondo del video. 'Calderone' = comportamento originale "
             "(Transizione Master1/2 + Calderone, eventuale Calderone 2). 'Foto Fissa'/'Video' "
             "sostituiscono tutto questo con un'unica immagine o un video in loop per tutta "
             "la durata — Master1/2 e Calderone 2 in quel caso non si applicano.")
    bg_static_file = None
    bg_video_file  = None
    if bg_source == "Foto Fissa":
        bg_static_file = st.file_uploader("Foto sfondo", type=["jpg","png","jpeg"], key="bg_static_file")
    elif bg_source == "Video":
        bg_video_file = st.file_uploader("Video sfondo", type=["mp4","mov","avi","mkv"], key="bg_video_file")
    st.divider()

    st.caption("🔀 Transizione (dissolvenza inizio/fine)")
    up_m1 = st.file_uploader("MASTER 1 — inizio", type=["jpg","png","jpeg"])
    up_m2 = st.file_uploader("MASTER 2 — fine",   type=["jpg","png","jpeg"])
    st.divider()
    up_t = st.file_uploader("CALDERONE", type=["jpg","png","jpeg"], accept_multiple_files=True)
    st.divider()

    calderone2_on = st.toggle("🖇️ Calderone 2", value=False, key="calderone2_on",
        help="Un secondo pool di foto, con lo STESSO trattamento glitch del Calderone principale "
             "(stesse impostazioni di Chaos/Speed/Geometria). Si mescola gradualmente col Calderone 1 "
             "dal punto che imposti sotto — stesso meccanismo di dissolvenza di Master 1/2.")
    calderone2_cfg = {'files': None, 'mix_from': 0.5}
    if calderone2_on:
        calderone2_cfg['files'] = st.file_uploader("Foto — Calderone 2", type=["jpg","png","jpeg"],
            accept_multiple_files=True, key="calderone2_files")
        calderone2_cfg['mix_from'] = st.slider("Calderone 2 subentra da (%)", 0, 100, 50,
            key="calderone2_mix_from",
            help="Prima di questo punto si vede solo il Calderone 1. Dopo, la probabilità di "
                 "pescare dal Calderone 2 cresce gradualmente fino al 100%.") / 100.0
        st.divider()

    up_a = st.file_uploader("AUDIO", type=["mp3","wav"])

    st.divider()
    preview_slot = st.container()

with bottom_container:
    # ---- STRISCE SELETTIVE ----
    dur = st.session_state.get("dur_input", 10)                          # prima di c3
    fmt_value = st.session_state.get("format_select", "16:9 (Orizzontale)")  # prima di c3

    overlays_cfg = []

    col_s, col_l = st.columns(2)
    with col_s:
        stripe_mode = st.toggle("🎯 Strisce Selettive", value=False, key="stripe_mode_g",
            help="Sfondo + finestre che mostrano il Calderone in movimento.")

        stripes            = []
        stripe_orientation = "Orizzontale"
        stripe_bg          = "Master 1"
        stripe_glitch      = False
        stripe_reverse     = False
        stripe_force_beat_react = False

        if stripe_mode:
            stripe_orientation = st.radio("Orientamento strisce",
                ["Orizzontale", "Verticale", "Mix H+V"], horizontal=True, key="stripe_orientation_g",
                help="Mix H+V: strisce pari=orizzontali, dispari=verticali")

            # scelta sfondo
            bg_opts = []
            if up_m1: bg_opts.append("Master 1")
            if up_m2: bg_opts.append("Master 2")
            bg_opts.append("Calderone")
            bg_opts.append("Render")
            stripe_bg = st.radio("🖼️ Sfondo", bg_opts, horizontal=True, key="stripe_bg_g",
                help="Master 1/2 = foto ferma. Calderone = foto in movimento. Render = glitch principale come sfondo, strisce sopra.")

            col_tog1, col_tog2 = st.columns(2)
            with col_tog1:
                stripe_glitch = st.toggle("⚡ Striscia glitchata", value=False, key="stripe_glitch_g",
                    help="OFF = striscia pulita. ON = glitchata.")
            with col_tog2:
                stripe_reverse = st.toggle("🔄 Reverse", value=False, key="stripe_reverse_g",
                    help="Inverte: strisce ferme, tutto il resto Calderone.")

            stripe_force_beat_react = st.toggle("🎵 Tutto a tempo", value=False, key="stripe_force_beat_g",
                help="Forza 'Sincronizza al beat' su TUTTE le strisce, ignorando il toggle di ognuna. "
                     "Utile per accendere/spegnere in blocco senza doverle aprire una per una.")

            st.divider()

            # --- Pulsanti aggiungi / gestione lista strisce ---
            col_add, col_ninfo = st.columns([1, 2])
            with col_add:
                if st.button("➕ Aggiungi striscia", key="add_stripe"):
                    st.session_state.stripe_ids.append(st.session_state.stripe_next_id)
                    st.session_state.stripe_next_id += 1
            with col_ninfo:
                st.caption(f"{len(st.session_state.stripe_ids)} striscia/e attive")

            # --- LOOP STRISCE: ogni striscia in un expander ---
            _to_delete = None

            for _loop_idx, i in enumerate(list(st.session_state.stripe_ids)):
                if _to_delete == i:
                    continue

                # Titolo expander dinamico
                _cur_orient = st.session_state.get(f"so_{i}", "Orizzontale")
                _kf_count = sum(len(v) for v in st.session_state.get(f"kf_stripe_{i}", {}).values())
                _kf_tag = f" · {_kf_count} KF" if _kf_count else ""
                _exp_title = f"Striscia {_loop_idx+1} — {_cur_orient}{_kf_tag}"

                with st.expander(_exp_title, expanded=(_loop_idx == 0), key=f"exp_stripe_{i}"):

                    # Pulsante elimina in cima all'expander
                    if st.button(f"🗑️ Elimina striscia {_loop_idx+1}", key=f"del_stripe_{i}"):
                        _to_delete = i
                        continue

                    # Forma
                    orient_opts = ["Orizzontale", "Verticale", "Striscia Ruotata", "Lancetta", "Cerchio"]
                    if f"so_{i}" not in st.session_state:
                        # Default assegnato UNA SOLA VOLTA alla creazione della striscia:
                        # le modifiche successive al toggle globale "Orientamento strisce"
                        # non devono più toccare strisce già esistenti.
                        if stripe_orientation == "Mix H+V":
                            st.session_state[f"so_{i}"] = "Orizzontale" if _loop_idx % 2 == 0 else "Verticale"
                        elif stripe_orientation in orient_opts:
                            st.session_state[f"so_{i}"] = stripe_orientation
                        else:
                            st.session_state[f"so_{i}"] = "Orizzontale"
                    stripe_orient_i = st.radio(
                        "Forma", orient_opts,
                        horizontal=True, key=f"so_{i}")

                    s_dict = {
                        'orientation':   stripe_orient_i,
                        'chroma_amount': 6,
                        'flash':         False,
                        'blend_mode':    'Normal',
                        'opacity':       1.0,
                    }

                    st.divider()

                    # Controlli specifici per tipo
                    if stripe_orient_i in ["Orizzontale", "Verticale"]:
                        ca, cb, cc = st.columns(3)
                        with ca:
                            s_dict['center'] = st.slider("Centro (%)", 0, 100, min(20+i*25,95), key=f"sc_{i}")
                        with cb:
                            s_dict['size'] = st.slider("Spessore (%)", 1, 100, 10, key=f"ss_{i}")
                        with cc:
                            s_dict['length'] = float(st.slider("Lunghezza (%)", 5, 100, 100, key=f"sl_{i}"))
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            s_dict['length_audio'] = st.toggle("Lunghezza reattiva", value=False, key=f"la_{i}")
                        with col_m2:
                            s_dict['move_random'] = st.toggle("Movimento random", value=False, key=f"mr_{i}")
                        s_dict['move_speed'] = 1.0
                        if s_dict['move_random']:
                            s_dict['move_speed'] = st.slider("Velocità movimento", 0.1, 5.0, 1.0, step=0.1, key=f"ms_{i}")
                        s_dict['beat_react'] = st.toggle("🎵 Sincronizza al beat", value=False, key=f"mbr_{i}",
                            help="Autonomo: senza 'Movimento random', la striscia resta ferma e lampeggia a tempo (piena visibilità sul colpo, invisibile altrimenti — segue solo il BPM, indipendentemente dal genere scelto). Se 'Movimento random' è attivo, invece è il movimento ad accelerare sul beat.")
                        s_dict['offset_length'] = float(st.slider("Offset dx/sx (%)", 0, 100, 50, key=f"oc_{i}"))

                    elif stripe_orient_i == "Striscia Ruotata":
                        col_cx, col_cy = st.columns(2)
                        with col_cx:
                            s_dict['cx'] = float(st.slider("Centro X (%)", 0, 100, 50, key=f"rcx_{i}"))
                        with col_cy:
                            s_dict['cy'] = float(st.slider("Centro Y (%)", 0, 100, 50, key=f"rcy_{i}"))
                        col_a, col_sp, col_l = st.columns(3)
                        with col_a:
                            s_dict['angle'] = float(st.slider("Angolo (°)", 0, 360, 0, key=f"rang_{i}",
                                help="0=orizzontale, 45=diagonale, 90=verticale"))
                        with col_sp:
                            s_dict['size'] = float(st.slider("Spessore (%)", 1, 100, 15, key=f"rsp_{i}"))
                        with col_l:
                            s_dict['length'] = float(st.slider("Lunghezza (%)", 5, 150, 100, key=f"rl_{i}",
                                help=">100 = esce dai bordi"))
                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                            s_dict['auto_rotate'] = st.toggle("Rotazione automatica", value=False, key=f"rar_{i}")
                        with col_r2:
                            s_dict['length_audio'] = st.toggle("Lunghezza reattiva", value=False, key=f"la_{i}")
                        s_dict['rotate_speed'] = 30.0
                        if s_dict['auto_rotate']:
                            s_dict['rotate_speed'] = st.slider("Velocità rotazione (°/sec)", 5.0, 360.0, 30.0, key=f"rrs_{i}")
                        s_dict['beat_react'] = st.toggle("🎵 Sincronizza al beat", value=False, key=f"rbr_{i}",
                            help="Autonomo: senza 'Rotazione automatica', la striscia resta ferma e lampeggia a tempo (piena visibilità sul colpo, invisibile altrimenti — segue solo il BPM, indipendentemente dal genere scelto). Se 'Rotazione automatica' è attiva, invece è la rotazione ad accelerare sul beat.")

                    elif stripe_orient_i == "Lancetta":
                        col_cx, col_cy = st.columns(2)
                        with col_cx:
                            s_dict['cx'] = float(st.slider("Pivot X (%)", 0, 100, 50, key=f"lcx_{i}"))
                        with col_cy:
                            s_dict['cy'] = float(st.slider("Pivot Y (%)", 0, 100, 50, key=f"lcy_{i}"))
                        col_a, col_l, col_t = st.columns(3)
                        with col_a:
                            s_dict['angle'] = float(st.slider("Angolo (°)", 0, 360, 90, key=f"lang_{i}",
                                help="0=destra, 90=su, 180=sinistra, 270=giù"))
                        with col_l:
                            s_dict['length'] = float(st.slider("Lunghezza (%)", 5, 100, 50, key=f"ll_{i}"))
                        with col_t:
                            s_dict['size'] = st.slider("Spessore (px)", 2, 100, 15, key=f"lt_{i}")
                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                            s_dict['auto_rotate'] = st.toggle("Rotazione automatica", value=False, key=f"lar_{i}")
                        with col_r2:
                            s_dict['length_audio'] = st.toggle("Lunghezza reattiva", value=False, key=f"la_{i}")
                        s_dict['rotate_speed'] = 30.0
                        if s_dict['auto_rotate']:
                            s_dict['rotate_speed'] = st.slider("Velocità rotazione (°/sec)", 5.0, 360.0, 30.0, key=f"lrs_{i}")
                        s_dict['beat_react'] = st.toggle("🎵 Sincronizza al beat", value=False, key=f"lbr_{i}",
                            help="Autonomo: senza 'Rotazione automatica', la lancetta resta ferma e lampeggia a tempo (piena visibilità sul colpo, invisibile altrimenti — segue solo il BPM, indipendentemente dal genere scelto). Se 'Rotazione automatica' è attiva, invece è la rotazione ad accelerare sul beat.")

                    elif stripe_orient_i == "Cerchio":
                        col_cx, col_cy = st.columns(2)
                        with col_cx:
                            s_dict['cx'] = float(st.slider("Centro X (%)", 0, 100, 50, key=f"ccx_{i}"))
                        with col_cy:
                            s_dict['cy'] = float(st.slider("Centro Y (%)", 0, 100, 50, key=f"ccy_{i}"))
                        col_r, col_t = st.columns(2)
                        with col_r:
                            s_dict['radius'] = float(st.slider("Raggio (%)", 1, 100, 30, key=f"cr_{i}"))
                        with col_t:
                            s_dict['size'] = st.slider("Spessore bordo (px)", 1, 50, 8, key=f"ct_{i}",
                                help="Usato solo se Cerchio pieno è OFF")
                        col_c1, col_c2, col_c3 = st.columns(3)
                        with col_c1:
                            s_dict['filled'] = st.toggle("Cerchio pieno", value=True, key=f"cf_{i}")
                        with col_c2:
                            s_dict['length_audio'] = st.toggle("Raggio reattivo", value=False, key=f"la_{i}",
                                help="Il raggio pulsa col volume")
                        with col_c3:
                            s_dict['auto_expand'] = st.toggle("Espansione ciclica", value=False, key=f"ce_{i}",
                                help="Il cerchio cresce e riparte dal centro")
                        s_dict['expand_speed'] = 20.0
                        if s_dict.get('auto_expand'):
                            s_dict['expand_speed'] = st.slider("Velocità espansione (%/sec)", 5.0, 100.0, 20.0, key=f"ces_{i}")
                        s_dict['beat_react'] = st.toggle("🎵 Sincronizza al beat", value=False, key=f"cbr_{i}",
                            help="Autonomo: senza 'Espansione ciclica', il cerchio resta fermo e lampeggia a tempo (piena visibilità sul colpo, invisibile altrimenti — segue solo il BPM, indipendentemente dal genere scelto). Se 'Espansione ciclica' è attiva, invece è l'espansione ad accelerare sul beat.")

                    st.divider()

                    # Effetti comuni
                    col_e1, col_e2 = st.columns(2)
                    with col_e1:
                        chroma_on = st.toggle("🌈 Chroma", value=False, key=f"ch_{i}")
                        s_dict['chroma_on'] = chroma_on
                    with col_e2:
                        s_dict['flash'] = st.toggle("⚡ Flash beat", value=False, key=f"fl_{i}")
                    if chroma_on:
                        s_dict['chroma_amount'] = st.slider("Intensità chroma (px)", 1, 30, 6, key=f"ca_{i}")
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        s_dict['blend_mode'] = st.selectbox("Blend mode",
                            ["Normal", "Screen", "Multiply", "Difference"], key=f"bm_{i}")
                    with col_b2:
                        s_dict['opacity'] = st.slider("Opacità (%)", 0, 100, 100, key=f"op_{i}") / 100.0

                    st.divider()

                    # Keyframe
                    _params_meta = []
                    if stripe_orient_i in ['Orizzontale', 'Verticale']:
                        _params_meta = [
                            {'param':'center','label':'Centro (%)','min_v':0,'max_v':100,'default_v':s_dict.get('center',50),'step_v':1},
                            {'param':'size','label':'Spessore (%)','min_v':1,'max_v':100,'default_v':s_dict.get('size',10),'step_v':1},
                            {'param':'length','label':'Lunghezza (%)','min_v':5,'max_v':100,'default_v':s_dict.get('length',100),'step_v':1},
                            {'param':'opacity','label':'Opacità','min_v':0.0,'max_v':1.0,'default_v':s_dict.get('opacity',1.0),'step_v':0.01},
                        ]
                    elif stripe_orient_i == 'Striscia Ruotata':
                        _params_meta = [
                            {'param':'angle','label':'Angolo (°)','min_v':0,'max_v':360,'default_v':s_dict.get('angle',0),'step_v':1},
                            {'param':'size','label':'Spessore (%)','min_v':1,'max_v':100,'default_v':s_dict.get('size',15),'step_v':1},
                            {'param':'length','label':'Lunghezza (%)','min_v':5,'max_v':150,'default_v':s_dict.get('length',100),'step_v':1},
                            {'param':'opacity','label':'Opacità','min_v':0.0,'max_v':1.0,'default_v':s_dict.get('opacity',1.0),'step_v':0.01},
                        ]
                    elif stripe_orient_i == 'Lancetta':
                        _params_meta = [
                            {'param':'angle','label':'Angolo (°)','min_v':0,'max_v':360,'default_v':s_dict.get('angle',90),'step_v':1},
                            {'param':'length','label':'Lunghezza (%)','min_v':5,'max_v':100,'default_v':s_dict.get('length',50),'step_v':1},
                            {'param':'size','label':'Spessore (px)','min_v':2,'max_v':100,'default_v':s_dict.get('size',15),'step_v':1},
                            {'param':'opacity','label':'Opacità','min_v':0.0,'max_v':1.0,'default_v':s_dict.get('opacity',1.0),'step_v':0.01},
                        ]
                    elif stripe_orient_i == 'Cerchio':
                        _params_meta = [
                            {'param':'radius','label':'Raggio (%)','min_v':1,'max_v':100,'default_v':s_dict.get('radius',30),'step_v':1},
                            {'param':'size','label':'Spessore bordo (px)','min_v':1,'max_v':50,'default_v':s_dict.get('size',8),'step_v':1},
                            {'param':'opacity','label':'Opacità','min_v':0.0,'max_v':1.0,'default_v':s_dict.get('opacity',1.0),'step_v':0.01},
                        ]
                    if _params_meta:
                        s_dict['keyframes'] = kf_expander_ui(i, dur, _params_meta)

                stripes.append(s_dict)

            if stripe_force_beat_react:
                for _s in stripes:
                    _s['beat_react'] = True

            # Applica l'eliminazione dopo aver renderizzato tutto
            if _to_delete is not None:
                st.session_state.stripe_ids.remove(_to_delete)
                st.rerun()

            st.divider()

            # --- EFFETTI GLOBALI STRISCIA ---
            col_fx1, col_fx2 = st.columns(2)
            with col_fx1:
                stripe_chroma = st.toggle("🌈 Chroma (globale)", value=False, key="stripe_chroma_g",
                    help="Aberrazione cromatica su tutte le strisce")
            with col_fx2:
                stripe_flash = st.toggle("⚡ Flash beat (globale)", value=False, key="stripe_flash_g",
                    help="Tutte le strisce lampeggiano sui beat forti")

            st.divider()

            # --- PRESET ---
            st.caption("💾 Preset strisce")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                preset_data = {
                    "stripe_orientation": stripe_orientation,
                    "stripe_bg":          stripe_bg,
                    "stripe_glitch":      stripe_glitch,
                    "stripe_reverse":     stripe_reverse,
                    "stripe_chroma":      stripe_chroma,
                    "stripe_flash":       stripe_flash,
                    "stripes":            stripes,
                }
                st.download_button(
                    "📤 Esporta preset",
                    data=json.dumps(preset_data, indent=2),
                    file_name="stripe_preset.json",
                    mime="application/json",
                )
            with col_p2:
                uploaded_preset = st.file_uploader("📥 Carica preset", type=["json"], key="preset_upload")
                if uploaded_preset and st.session_state.get("_last_preset_file") != uploaded_preset.file_id:
                    try:
                        loaded = json.load(uploaded_preset)
                        st.session_state.preset = loaded
                        st.session_state["_last_preset_file"] = uploaded_preset.file_id

                        # --- Ripristina i toggle/radio globali (via le loro key esplicite) ---
                        if "stripe_orientation" in loaded:
                            st.session_state["stripe_orientation_g"] = loaded["stripe_orientation"]
                        if "stripe_bg" in loaded:
                            st.session_state["stripe_bg_g"] = loaded["stripe_bg"]
                        if "stripe_glitch" in loaded:
                            st.session_state["stripe_glitch_g"] = loaded["stripe_glitch"]
                        if "stripe_reverse" in loaded:
                            st.session_state["stripe_reverse_g"] = loaded["stripe_reverse"]
                        st.session_state["stripe_mode_g"] = True  # riapri il pannello strisce

                        # --- Ripristina gli ID strisce in base al numero salvato nel preset ---
                        loaded_stripes = loaded.get("stripes", [])
                        n_loaded = len(loaded_stripes)
                        if n_loaded > 0:
                            st.session_state.stripe_ids = list(range(n_loaded))
                            st.session_state.stripe_next_id = n_loaded

                        # --- Ripristina ogni striscia nei widget corrispondenti (per key) ---
                        SHAPE_KEY_MAP = {
                            "Orizzontale": {"center": "sc", "size": "ss", "length": "sl",
                                            "length_audio": "la", "move_random": "mr", "move_speed": "ms",
                                            "beat_react": "mbr", "offset_length": "oc"},
                            "Verticale": {"center": "sc", "size": "ss", "length": "sl",
                                          "length_audio": "la", "move_random": "mr", "move_speed": "ms",
                                          "beat_react": "mbr", "offset_length": "oc"},
                            "Striscia Ruotata": {"cx": "rcx", "cy": "rcy", "angle": "rang", "size": "rsp",
                                                 "length": "rl", "auto_rotate": "rar", "length_audio": "la",
                                                 "rotate_speed": "rrs", "beat_react": "rbr"},
                            "Lancetta": {"cx": "lcx", "cy": "lcy", "angle": "lang", "length": "ll",
                                         "size": "lt", "auto_rotate": "lar", "length_audio": "la",
                                         "rotate_speed": "lrs", "beat_react": "lbr"},
                            "Cerchio": {"cx": "ccx", "cy": "ccy", "radius": "cr", "size": "ct",
                                        "filled": "cf", "length_audio": "la", "auto_expand": "ce",
                                        "expand_speed": "ces", "beat_react": "cbr"},
                        }
                        COMMON_KEY_MAP = {"chroma_on": "ch", "flash": "fl", "chroma_amount": "ca",
                                           "blend_mode": "bm"}

                        for idx, s in enumerate(loaded_stripes):
                            orient = s.get("orientation", "Orizzontale")
                            st.session_state[f"so_{idx}"] = orient
                            keymap = SHAPE_KEY_MAP.get(orient, SHAPE_KEY_MAP["Orizzontale"])
                            for field, prefix in keymap.items():
                                if field in s:
                                    st.session_state[f"{prefix}_{idx}"] = s[field]
                            for field, prefix in COMMON_KEY_MAP.items():
                                if field in s:
                                    st.session_state[f"{prefix}_{idx}"] = s[field]
                            if "opacity" in s:
                                st.session_state[f"op_{idx}"] = int(round(s["opacity"] * 100))
                            if "keyframes" in s:
                                st.session_state[f"kf_stripe_{idx}"] = s["keyframes"]

                        st.success("Preset caricato!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Errore preset: {e}")

        else:
            stripe_chroma = False
            stripe_flash  = False

    with col_l:
        overlay_panel_on = st.toggle("🎬 Overlay Foto/Video", value=False,
            key="overlay_panel_on",
            help="Foto o video posizionabili liberamente sullo sfondo — solo posizione e "
                 "dimensione (come i comandi X/Y delle strisce), nessun effetto/blend/pulsazione. "
                 "Se è un video più corto della durata del render, va in loop.")

        if 'overlay_ids' not in st.session_state: st.session_state.overlay_ids = []
        if 'overlay_next_id' not in st.session_state: st.session_state.overlay_next_id = 0

        if overlay_panel_on:
            col_oadd, col_oinfo = st.columns([1, 2])
            with col_oadd:
                if st.button("➕ Aggiungi overlay", key="add_overlay"):
                    st.session_state.overlay_ids.append(st.session_state.overlay_next_id)
                    st.session_state.overlay_next_id += 1
            with col_oinfo:
                st.caption(f"{len(st.session_state.overlay_ids)} overlay attivi")

            _to_delete_overlay = None

            for _oidx, oi in enumerate(list(st.session_state.overlay_ids)):
                if _to_delete_overlay == oi:
                    continue

                with st.expander(f"Overlay {_oidx+1}", expanded=(_oidx == 0), key=f"exp_overlay_{oi}"):
                    if st.button(f"🗑️ Elimina overlay {_oidx+1}", key=f"del_overlay_{oi}"):
                        _to_delete_overlay = oi
                        continue

                    o_file = st.file_uploader("Foto o video",
                        type=["png", "jpg", "jpeg", "mp4", "mov", "avi", "mkv"], key=f"ov_file_{oi}")

                    col_ox, col_oy, col_oz = st.columns(3)
                    with col_ox:
                        o_cx = float(st.slider("Posizione X (%)", -100, 200, 50, key=f"ov_cx_{oi}",
                            help="50 = centrato. Sotto 0 o sopra 100 esce dal fotogramma."))
                    with col_oy:
                        o_cy = float(st.slider("Posizione Y (%)", -100, 200, 50, key=f"ov_cy_{oi}"))
                    with col_oz:
                        o_scale = st.slider("Dimensione", 0.05, 2.0, 0.4, step=0.05, key=f"ov_sc_{oi}",
                            help="1.0 = riempie il fotogramma (contain-fit).")

                    overlays_cfg.append({'file': o_file, 'cx': o_cx, 'cy': o_cy, 'scale': o_scale})

            if _to_delete_overlay is not None:
                st.session_state.overlay_ids.remove(_to_delete_overlay)
                st.rerun()

    with preview_slot:
        st.caption("🔍 Anteprima")

        if bg_source == "Video":
            st.caption("🎬 Sfondo Video: si vede solo nel render finale, non nell'anteprima statica.")

        prev_choices = []
        prev_files   = {}
        if bg_source == "Foto Fissa" and bg_static_file:
            prev_choices.append("Sfondo (Foto Fissa)"); prev_files["Sfondo (Foto Fissa)"] = bg_static_file
        if up_m1: prev_choices.append("Master 1");             prev_files["Master 1"] = up_m1
        if up_m2: prev_choices.append("Master 2");             prev_files["Master 2"] = up_m2
        if up_t:  prev_choices.append("Prima foto Calderone"); prev_files["Prima foto Calderone"] = up_t[0]

        if not prev_choices:
            st.caption("Carica almeno una foto (Master, Calderone o Sfondo) per vedere l'anteprima.")
        else:
            prev_sel = st.selectbox("Anteprima su", prev_choices,
                label_visibility="collapsed", key="unified_prev_sel")
            pf = prev_files[prev_sel]
            pf.seek(0)
            prev_img_full = np.array(Image.open(pf).convert("RGB"))

            _fmt_dims = {"16:9 (Orizzontale)": (1280, 720),
                         "9:16 (Verticale)":  (720, 1280),
                         "1:1 (Quadrato)":    (1080, 1080)}
            _fw, _fh = _fmt_dims.get(fmt_value, (1280, 720))
            st.caption(f"Formato: {fmt_value}")
            prev_img_cropped = cover_crop(prev_img_full, _fw, _fh)  # stesso ritaglio del render

            ph, pw   = prev_img_cropped.shape[:2]
            pscale   = 190 / max(ph, pw)  # anteprima compatta
            pdw, pdh = int(pw * pscale), int(ph * pscale)
            preview_out = cv2.resize(prev_img_cropped, (pdw, pdh)).copy()

            _capt = "Anteprima"
            if stripe_mode and stripes:
                draw_stripe_preview_overlay(preview_out, stripes, stripe_orientation, pdh, pdw)
                _capt += " · viola = striscia attiva"
                if stripe_reverse:
                    _capt += " · REVERSE ON"

            if overlay_panel_on and overlays_cfg:
                _ov_preview = []
                for _ov in overlays_cfg:
                    if _ov['file'] is None:
                        continue
                    _ov_name = _ov['file'].name.lower()
                    if _ov_name.endswith(('.mp4', '.mov', '.avi', '.mkv')):
                        # placeholder: un video richiede decodifica, qui mostro solo dove/quanto sarà grande
                        _ph = np.full((200, 200, 4), [90, 90, 90, 210], dtype=np.uint8)
                    else:
                        _ph = load_overlay_image(_ov['file'])
                    _ov_preview.append({'rgba': _ph, 'scale': _ov['scale'], 'cx': _ov['cx'], 'cy': _ov['cy']})
                if _ov_preview:
                    preview_out = apply_media_overlays(preview_out, _ov_preview, pdh, pdw)
                    _capt += " · overlay posizionati"

            st.image(preview_out, caption=_capt, use_container_width=True)

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
    with st.expander("✨ Effetti globali sul render (chroma, flash beat)", expanded=False):
        # --- EFFETTI GLOBALI — disponibili sempre, anche senza strisce selettive ---
        col_gfx1, col_gfx2 = st.columns(2)
        with col_gfx1:
            global_chroma = st.toggle("🌈 Chroma aberration", value=False, key="global_chroma",
                help="Aberrazione cromatica su tutto il frame (o su tutte le strisce se attive)")
            global_chroma_amt = 6
            if global_chroma:
                global_chroma_amt = st.slider("Intensità chroma (px)", 1, 30, 6, key="global_chroma_amt",
                    help="Quanti pixel sfasa i canali R e B")
            if stripe_mode and global_chroma:
                stripe_chroma = True
        with col_gfx2:
            global_flash = st.toggle("⚡ Flash beat", value=False, key="global_flash",
                help="Il frame lampeggia sui beat forti (richiede audio).")
            global_flash_threshold = 0.7
            global_flash_intensity = 100
            if global_flash:
                global_flash_threshold = st.slider("Soglia beat (%)", 0, 100, 70, key="global_flash_thr",
                    help="Quanto deve essere forte il beat per far scattare il flash. 0=sempre, 100=solo sui picchi massimi.") / 100.0
                global_flash_intensity = st.slider("Intensità flash (%)", 0, 100, 100, key="global_flash_int",
                    help="100=frame bianco/pulito totale, valori bassi=flash parziale miscelato")
            if stripe_mode and global_flash:
                stripe_flash = True

    st.divider()
    seq_mode = st.toggle("🔢 Sequenza Ordinata", value=False,
        help="Le foto del Calderone vengono usate in ordine (1→2→3…) invece che random.")

with c3:
    st.subheader("🎬 Rendering")
    fmt = st.selectbox("Format", ["16:9 (Orizzontale)", "9:16 (Verticale)", "1:1 (Quadrato)"],
        key="format_select")
    dur = st.number_input("Durata (sec)", 1, 300, 10, key="dur_input")

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

    beat_sync = st.toggle("🎵 A tempo di musica", value=False,
        help="Attiva solo se hai caricato un audio. Disabilitato in modalità Slideshow.")
    genre = "Techno / House"
    manual_bpm = None
    onset_sensitivity = None
    if beat_sync:
        genre = st.selectbox("Genere", list(GENRE_PRESETS.keys()))

        bpm_mode = st.radio("🎯 BPM", ["Rileva automaticamente", "Inserisci manualmente"],
            horizontal=True, key="bpm_mode_radio",
            help="Se il detect sbaglia (es. tracce con poca batteria), scrivi tu il BPM.")
        if bpm_mode == "Inserisci manualmente":
            manual_bpm = st.number_input("BPM", min_value=40, max_value=220, value=120, step=1,
                key="manual_bpm_input")

        onset_sensitivity = st.slider(
            "🎚️ Sensibilità Onset", 0.0, 1.0, GENRE_PRESETS[genre]["onset"] / 100.0, step=0.05,
            help="Individua ogni singolo transiente audio, regolare o no. Basso = solo colpi forti. Alto = intercetta anche i transienti deboli."
        )

        # --- anteprima BPM + grafico beat/onset: stessa analisi della generazione finale ---
        if slideshow_mode:
            st.caption("⚠️ Beat Sync disattivato in modalità Slideshow: l'anteprima qui sotto non si applicherà al render.")
        if up_a is not None:
            preview_key = f"{up_a.name}_{up_a.size}_{round(dur,2)}_{genre}_{manual_bpm}_{round(onset_sensitivity,2)}_{slideshow_mode}"
            if st.session_state.get("audio_preview_key") != preview_key:
                with st.spinner("🎯 Analisi audio in corso..."):
                    y_prev, sr_prev = get_or_decode_audio(up_a, dur)
                    total_f_prev = int(dur * 24)
                    preview_result = analyze_audio(
                        y_prev, sr_prev, total_f_prev, 24,
                        True, slideshow_mode, genre, manual_bpm, onset_sensitivity
                    )
                    st.session_state["audio_preview"] = preview_result
                    st.session_state["audio_preview_key"] = preview_key
            preview_result = st.session_state["audio_preview"]

            show_audio_info = st.checkbox("📊 Mostra dettagli audio (BPM rilevato, grafico beat/onset)",
                value=False, key="show_audio_info")
            if show_audio_info:
                if preview_result['detected_bpm'] > 0:
                    st.info(f"🎯 BPM {preview_result['bpm_source'].lower()}: **{preview_result['detected_bpm']:.1f}**")
                    st.caption("Ti sembra raddoppiato o dimezzato rispetto al vero tempo del brano? Correggilo qui:")
                    col_half, col_double = st.columns(2)
                    with col_half:
                        if st.button("÷2 BPM", key="bpm_half_btn", use_container_width=True):
                            new_bpm = int(round(preview_result['detected_bpm'] / 2.0))
                            st.session_state["bpm_mode_radio"] = "Inserisci manualmente"
                            st.session_state["manual_bpm_input"] = max(40, min(220, new_bpm))
                            st.rerun()
                    with col_double:
                        if st.button("×2 BPM", key="bpm_double_btn", use_container_width=True):
                            new_bpm = int(round(preview_result['detected_bpm'] * 2.0))
                            st.session_state["bpm_mode_radio"] = "Inserisci manualmente"
                            st.session_state["manual_bpm_input"] = max(40, min(220, new_bpm))
                            st.rerun()

                total_f_prev = len(preview_result['audio_envelope'])
                n_points = min(400, total_f_prev)
                idx = np.linspace(0, total_f_prev - 1, n_points).astype(int)
                chart_df = pd.DataFrame({
                    "Volume": preview_result['audio_envelope'][idx],
                    "Beat":   preview_result['beat_envelope'][idx],
                    "Onset":  preview_result['onset_envelope'][idx],
                })
                st.caption("📊 Anteprima: volume, beat e onset rilevati lungo la durata")
                st.line_chart(chart_df, height=180)
        else:
            st.caption("Carica un audio per l'anteprima BPM e il grafico beat/onset.")

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
            stripe_bg, stripe_glitch, stripe_reverse,
            stripe_chroma, stripe_flash,
            global_chroma, global_chroma_amt,
            global_flash, global_flash_threshold, global_flash_intensity,
            manual_bpm=manual_bpm, onset_sensitivity=onset_sensitivity,
            calderone2_cfg=calderone2_cfg,
            bg_source=bg_source, bg_static_file=bg_static_file, bg_video_file=bg_video_file,
            overlays_cfg=overlays_cfg
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

        # --- EXPORT FRAME SINGOLO ---
        if stripe_mode:
            st.divider()
            st.caption("🖼️ Export frame singolo")
            col_fr1, col_fr2 = st.columns([2,1])
            with col_fr1:
                frame_sec = st.slider("Secondo da esportare", 0.0, float(dur), 0.0, step=0.1)
            with col_fr2:
                if st.button("📸 Estrai frame"):
                    cap = cv2.VideoCapture(st.session_state.v_path)
                    cap.set(cv2.CAP_PROP_POS_MSEC, frame_sec * 1000)
                    ret, frame_bgr = cap.read()
                    cap.release()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        pil_frame = Image.fromarray(frame_rgb)
                        tmp_png = tempfile.mktemp(suffix=".png")
                        pil_frame.save(tmp_png)
                        st.session_state.frame_export = tmp_png
            if st.session_state.frame_export and os.path.exists(st.session_state.frame_export):
                st.image(st.session_state.frame_export, use_container_width=True)
                st.download_button("💾 Scarica PNG",
                    open(st.session_state.frame_export, "rb"),
                    file_name=f"{base}_frame_{frame_sec:.1f}s.png")

        if st.session_state.r_path:
            with open(st.session_state.r_path, "r") as f: r_txt = f.read()
            st.text_area("📄 TECHNICAL REPORT", r_txt, height=380)
            st.download_button("📄 SCARICA REPORT", r_txt,
                file_name=f"{base}_report.txt")
