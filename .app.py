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

# --- PRESET GENERE ---
GENRE_PRESETS = {
    "Techno / House":  {"beat_strength": 70, "beat_decay": 20, "onset":  0, "cache": 20, "rhythm": False},
    "Orchestrale":     {"beat_strength": 20, "beat_decay": 80, "onset": 40, "cache": 25, "rhythm": True},
    "Pop / Soul":      {"beat_strength": 50, "beat_decay": 50, "onset": 60, "cache": 35, "rhythm": False},
    "Glitch / Noise":  {"beat_strength": 90, "beat_decay": 10, "onset": 80, "cache": 70, "rhythm": True},
    "Drone / Pad":     {"beat_strength":  0, "beat_decay": 60, "onset": 20, "cache": 15, "rhythm": True},
    "Hip Hop / Jazz":  {"beat_strength": 60, "beat_decay": 35, "onset": 50, "cache": 30, "rhythm": False},
}

# Config condivisa per le strisce con moto "a rampa" (angolo/raggio che cresce nel tempo o
# scatta sul beat). Orizzontale/Verticale non è qui perché il suo moto automatico è un'oscillazione
# sinusoidale, non una rampa, e vive nel proprio ramo in generate_master.
STRIPE_MOTION_CONFIG = {
    "Lancetta":         dict(auto_key="auto_rotate", speed_key="rotate_speed", speed_default=30.0,
                             base_key="angle", base_default=90.0, wrap_hi=360.0,
                             add_base_to_ramp=True, pulse_amp=45.0, pulse_wrap=True),
    "Striscia Ruotata": dict(auto_key="auto_rotate", speed_key="rotate_speed", speed_default=30.0,
                             base_key="angle", base_default=0.0, wrap_hi=360.0,
                             add_base_to_ramp=True, pulse_amp=45.0, pulse_wrap=True),
    "Cerchio":          dict(auto_key="auto_expand", speed_key="expand_speed", speed_default=20.0,
                             base_key="radius", base_default=30.0, wrap_hi=100.0,
                             add_base_to_ramp=False, pulse_amp=30.0, pulse_wrap=False,
                             pulse_lo=1.0, pulse_hi=100.0),
}


def resolve_reactive_val(s_dict, base_val, offset, auto_key):
    """
    Sceglie tra valore statico (base_val) e offset animato (rampa continua o scatto sul beat),
    a seconda che 'auto_key' (es. auto_rotate/auto_expand/move_random) o 'beat_react' sia attivo
    per questa striscia. Usato per tenere allineati calcolo-traiettoria e disegno.
    """
    if s_dict.get(auto_key, False) or s_dict.get('beat_react', False):
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


def kf_expander_ui(stripe_id, dur, params_meta):
    """
    Un solo expander per striscia — tabella keyframe unificata.
    params_meta: lista di dict { param, label, min_v, max_v, default_v, step_v }
    Ritorna dict { param: [{t, v}, ...] }
    """
    state_key = f"kf_stripe_{stripe_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {}
    kf_state = st.session_state[state_key]
    for m in params_meta:
        kf_state.setdefault(m['param'], [])

    param_labels = [m['label'] for m in params_meta]
    param_map    = {m['param']: m for m in params_meta}

    total_kf = sum(len(v) for v in kf_state.values())
    exp_label = f"\U0001f3ac Keyframe striscia {stripe_id+1}" + (f"  \u2014 {total_kf} KF attivi" if total_kf else "")

    with st.expander(exp_label, expanded=False):
        st.caption("Imposta la striscia con gli slider sopra, poi aggiungi qui i cambiamenti nel tempo.")
        ca, cb, cc, cd = st.columns([2, 2, 2, 1])
        with ca:
            sel_label = st.selectbox("Parametro", param_labels, key=f"kf_sel_{stripe_id}")
            sel_param = next(m['param'] for m in params_meta if m['label'] == sel_label)
            meta = param_map[sel_param]
        with cb:
            new_t = st.number_input("t (sec)", min_value=0.0, max_value=float(max(dur, 1)),
                                    value=0.0, step=0.5, key=f"kf_t_{stripe_id}")
        with cc:
            new_v = st.number_input("valore", min_value=float(meta['min_v']),
                                    max_value=float(meta['max_v']),
                                    value=float(meta['default_v']),
                                    step=float(meta.get('step_v', 1)),
                                    key=f"kf_v_{stripe_id}")
        with cd:
            st.write("")
            st.write("")
            if st.button("\u2795 Aggiungi", key=f"kf_add_{stripe_id}"):
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
                    if st.button("\u2715", key=f"kf_del_{stripe_id}_{p}_{ki}"):
                        to_del = (p, ki)
            if to_del:
                p, ki = to_del
                kf_state[p].pop(ki)
                st.session_state[state_key] = kf_state
        else:
            st.caption("\u2014 Nessun KF: la striscia usa i valori degli slider per tutta la durata.")

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


def apply_stripe_window(bg_frame, calder_clean, calder_glitch, h, w,
                        stripes, stripe_orientation, stripe_glitch,
                        stripe_reverse=False, audio_envelope_val=1.0,
                        stripe_offsets=None,
                        stripe_chroma=False, stripe_flash=False,
                        beat_val=0.0,
                        t=0.0, total_dur=10.0):
    """
    stripes: lista di dict con keys: center, size, length, length_audio,
             move_random, move_speed, offset_length, chroma_amount,
             blend_mode, opacity
    stripe_chroma:  aberrazione cromatica dentro la striscia
    stripe_flash:   striscia si spegne (mostra bg) sui beat forti
    beat_val:       valore beat envelope corrente (0-1)
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

    def _paste_h(p0, p1, l0, l1, chroma_amt, mode, opacity):
        if p1 > p0 and l1 > l0:
            patch = src_stripe[p0:p1, l0:l1].copy()
            if stripe_chroma and chroma_amt > 0:
                patch = apply_chroma(patch, chroma_amt)
            base_patch = out[p0:p1, l0:l1]
            out[p0:p1, l0:l1] = blend_patch(base_patch, patch, mode, opacity)

    def _paste_v(p0, p1, l0, l1, chroma_amt, mode, opacity):
        if p1 > p0 and l1 > l0:
            patch = src_stripe[l0:l1, p0:p1].copy()
            if stripe_chroma and chroma_amt > 0:
                patch = apply_chroma(patch, chroma_amt)
            base_patch = out[l0:l1, p0:p1]
            out[l0:l1, p0:p1] = blend_patch(base_patch, patch, mode, opacity)

    def _draw(s, offset, is_h):
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
        dim = (h, w) if is_h else (w, h)
        p0, p1, l0, l1 = compute_stripe_coords(center, size, base_len, length_offset, dim)
        if is_h:
            _paste_h(p0, p1, l0, l1, chroma_amt, blend_mode, opacity)
        else:
            _paste_v(p0, p1, l0, l1, chroma_amt, blend_mode, opacity)

    for idx, s in enumerate(stripes):
        if flash_active:
            continue
        offset = stripe_offsets[idx]
        s_orient = s.get('orientation', stripe_orientation)
        chroma_amt = int(s.get('chroma_amount', 6))
        mode    = s.get('blend_mode', 'Normal')
        opacity = float(s.get('opacity', 1.0))
        chroma_on = stripe_chroma and chroma_amt > 0

        if s_orient == "Lancetta":
            # angolo base + rotazione automatica nel tempo (offset usato come angolo corrente)
            angle_base = kf_get(s, 'angle', t, total_dur, s.get('angle', 90.0))
            angle = resolve_reactive_val(s, angle_base, offset, 'auto_rotate')
            length_pct = kf_get(s, 'length', t, total_dur, s.get('length', 50.0))
            if s.get('length_audio', False):
                length_pct = length_pct * (0.2 + 0.8 * audio_envelope_val)
            opacity = kf_get(s, 'opacity', t, total_dur, float(s.get('opacity', 1.0)))
            draw_lancetta(out, src_stripe, h, w,
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
            draw_cerchio(out, src_stripe, h, w,
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
            draw_striscia_ruotata(out, src_stripe, h, w,
                                  s.get('cx', 50.0), s.get('cy', 50.0),
                                  angle,
                                  float(kf_get(s, 'size', t, total_dur, s.get('size', 15.0))),
                                  length_pct,
                                  chroma_on, chroma_amt, mode, opacity)

        elif s_orient in ("Orizzontale", "Verticale"):
            _draw(s, offset, s_orient == "Orizzontale")

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
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                detected_bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
                bpm_source = "AUTO"
                beat_times = librosa.frames_to_time(beat_frames, sr=sr)

            result['beat_times']   = beat_times
            result['detected_bpm'] = detected_bpm
            result['bpm_source']   = bpm_source

            decay_rate = 1.0 - (bd / 100.0) * 0.98
            for bt in beat_times:
                bf = int(bt * fps)
                for df in range(min(int(fps * 0.5), total_f - bf)):
                    result['beat_envelope'][bf + df] = max(result['beat_envelope'][bf + df], decay_rate ** df)

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
                    manual_bpm=None, onset_sensitivity=None):

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
    onset_envelope  = np.zeros(total_f)
    rhythm_envelope = None
    audio_peak      = 0.0
    bs, bd, op, bc  = 0, 50, 0, 30
    rhythm_tracking = False
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
        onset_envelope  = audio_result['onset_envelope']
        rhythm_envelope = audio_result['rhythm_envelope']
        audio_peak      = audio_result['audio_peak']
        bs, bd, op, bc  = audio_result['bs'], audio_result['bd'], audio_result['op'], audio_result['bc']
        rhythm_tracking = audio_result['rhythm_tracking']
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
        dt = 1.0 / fps
        # tempo "deformato": accelera sui beat/onset invece di scorrere costante
        beat_mult = 1.0 + beat_envelope * 3.0 + onset_envelope * 2.0
        warped_t = np.cumsum(beat_mult) * dt

        # scatto autonomo: ad ogni beat/onset la traiettoria "salta" invece di scorrere
        _trigger = (beat_envelope >= 0.999) | (onset_envelope > 0)
        _trigger_idx = np.cumsum(_trigger.astype(int))
        _pulse_sign = np.where(_trigger_idx % 2 == 0, 1.0, -1.0)
        _pulse_amt_raw = np.maximum(beat_envelope, onset_envelope)  # 0-1

        def _beat_pulse(base_val, amp, wrap=False, lo=0.0, hi=100.0):
            delta = _pulse_sign * _pulse_amt_raw * amp
            if wrap:
                return (base_val + delta) % hi
            return np.clip(base_val + delta, lo, hi)

        def _ramp_or_pulse(s, cfg, base_t, react):
            """Rampa continua (auto_key ON) oppure scatto sul beat (react ON, auto_key OFF)."""
            base_val = s.get(cfg["base_key"], cfg["base_default"])
            if s.get(cfg["auto_key"], False):
                spd = s.get(cfg["speed_key"], cfg["speed_default"])
                ramp = spd * base_t
                if cfg["add_base_to_ramp"]:
                    ramp = base_val + ramp
                return ramp % cfg["wrap_hi"]
            elif react:
                return _beat_pulse(base_val, amp=cfg["pulse_amp"], wrap=cfg["pulse_wrap"],
                                    lo=cfg.get("pulse_lo", 0.0), hi=cfg.get("pulse_hi", cfg["wrap_hi"]))
            else:
                return np.full(total_f, 50.0)

        for s in stripes:
            s_orient = s.get('orientation', stripe_orientation)
            react = bool(s.get('beat_react', False)) and beat_sync and not slideshow_mode
            base_t = warped_t if react else t_arr

            if s_orient in STRIPE_MOTION_CONFIG:
                traj = _ramp_or_pulse(s, STRIPE_MOTION_CONFIG[s_orient], base_t, react)
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
                elif react:
                    # autonomo: la striscia scatta sui beat/onset, senza bisogno di movimento random
                    traj = _beat_pulse(base_offset, amp=35.0, lo=0.0, hi=100.0)
                else:
                    traj = np.full(total_f, 50.0)
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

            def _get_bg_slide(glitched_frame=None):
                if stripe_use_render and glitched_frame is not None:
                    return glitched_frame
                return stripe_bg_static if stripe_bg_static is not None else img_next

            if cycle_pos < slide_hold:
                if stripe_mode and stripes:
                    out_frame = apply_stripe_window(_get_bg_slide(img_cur), img_cur, img_cur, h, w,
                                                    stripes, stripe_orientation, False,
                                                    stripe_reverse, _aenv, _soff,
                                                    stripe_chroma, stripe_flash, _bval,
                                                    t=t, total_dur=max_limit)
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
                    if stripe_mode and stripes:
                        out_frame = apply_stripe_window(_get_bg_slide(glitched), base, glitched, h, w,
                                                        stripes, stripe_orientation, stripe_glitch,
                                                        stripe_reverse, _aenv, _soff,
                                                        stripe_chroma, stripe_flash, _bval,
                                                        t=t, total_dur=max_limit)
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
                                                        stripe_chroma, stripe_flash, _bval,
                                                        t=t, total_dur=max_limit)
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
                        force = beat_sync and onset_envelope[f] > 0 and random.random() < (bc / 100.0) * onset_envelope[f]
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
                    force = beat_sync and onset_envelope[f] > 0 and random.random() < (bc / 100.0) * onset_envelope[f]
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

            _aenv = float(audio_envelope[f])
            _bval = float(beat_envelope[f])
            _soff = [stripe_offsets_t[si][f] for si in range(len(stripes))] if stripes else []

            if orientation == "Nessun Effetto":
                if stripe_mode and stripes:
                    calder_clean = pick()
                    if stripe_use_render:
                        # nessun glitch → frame pulito come sfondo
                        _bg = calder_clean
                    else:
                        _bg = stripe_bg_static if stripe_bg_static is not None else pick()
                    return cv2.resize(
                        apply_stripe_window(_bg, calder_clean, calder_clean, h, w,
                                            stripes, stripe_orientation, False, stripe_reverse,
                                            _aenv, _soff, stripe_chroma, stripe_flash, _bval,
                                            t=t, total_dur=max_limit),
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
            if stripe_mode and stripes:
                if stripe_use_render:
                    _bg = frame.copy()
                else:
                    _bg = stripe_bg_static if stripe_bg_static is not None else pick()
                frame = apply_stripe_window(_bg, calder_clean, frame, h, w,
                                            stripes, stripe_orientation, stripe_glitch,
                                            stripe_reverse, _aenv, _soff,
                                            stripe_chroma, stripe_flash, _bval,
                                            t=t, total_dur=max_limit)

            # --- Effetti globali standalone (senza strisce selettive) ---
            if not stripe_mode:
                if global_chroma:
                    frame = apply_chroma(frame, global_chroma_amt)
                if global_flash and _bval > global_flash_threshold:
                    alpha = global_flash_intensity / 100.0
                    frame = (calder_clean * (1.0 - alpha) + frame * (1.0 - alpha)).astype(np.uint8)

            result = cv2.resize(frame, (out_w, out_h))
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

    slide_info = ""
    if slideshow_mode:
        slide_info = f"""
* MODALITÀ: SLIDESHOW LENTO
* Durata foto: {slide_hold}s | Transizione: {slide_trans}s
* Tipo transizione: {slide_trans_type}
* Sequenza: {'ORDINATA' if seq_mode else 'RANDOM'}"""

    stripe_info = ""
    if stripe_mode and stripes:
        bg_label = "Frame" if stripe_bg == "Calderone" else stripe_bg
        stripe_info = f"""
* STRISCE SELETTIVE: {len(stripes)} striscia/e ({stripe_orientation})
* Sfondo: {bg_label} | Striscia: {'GLITCHATA' if stripe_glitch else 'ORIGINALE'}"""

    report_text = f"""[SLICE_PHOTO_DISSECTION] // VOL_01 // H.264 // DATA_FRAGMENT
:: MOTORE: recursive_cut_pro [v10.0 — keyframe]
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
* BPM: {f'{detected_bpm:.1f} ({bpm_source})' if beat_sync and not slideshow_mode and detected_bpm > 0 else 'N/A'}
* Onset Sensitivity: {f'{int(onset_sensitivity*100)}%' if beat_sync and not slideshow_mode and onset_sensitivity is not None else 'N/A (preset)'}
* Power Curve: {'BYPASSED' if rhythm_on else 'ON'}
* Sequenza Frame: {'ORDINATA' if seq_mode else 'RANDOM'}{slide_info}{stripe_info}

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

# dur viene definito in c3 ma serve in c2 (KF UI) — leggo da session_state con default
if 'dur_value' not in st.session_state:
    st.session_state.dur_value = 10

c1, c2, c3 = st.columns([1, 1.2, 1])

with c1:
    st.subheader("🖼️ Assets")
    up_m1 = st.file_uploader("MASTER 1 — inizio", type=["jpg","png","jpeg"])
    up_m2 = st.file_uploader("MASTER 2 — fine",   type=["jpg","png","jpeg"])
    st.divider()
    up_t = st.file_uploader("CALDERONE", type=["jpg","png","jpeg"], accept_multiple_files=True)
    st.divider()
    up_a = st.file_uploader("AUDIO", type=["mp3","wav"])

    st.divider()
    stripe_preview_slot = st.container()

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
    dur = st.session_state.dur_value  # disponibile per kf_ui prima di c3
    stripe_mode = st.toggle("🎯 Strisce Selettive", value=False,
        help="Sfondo + finestre che mostrano il Calderone in movimento.")

    stripes            = []
    stripe_orientation = "Orizzontale"
    stripe_bg          = "Master 1"
    stripe_glitch      = False
    stripe_reverse     = False
    stripe_force_beat_react = False

    if stripe_mode:
        stripe_orientation = st.radio("Orientamento strisce",
            ["Orizzontale", "Verticale", "Mix H+V"], horizontal=True,
            help="Mix H+V: strisce pari=orizzontali, dispari=verticali")

        # scelta sfondo
        bg_opts = []
        if up_m1: bg_opts.append("Master 1")
        if up_m2: bg_opts.append("Master 2")
        bg_opts.append("Calderone")
        bg_opts.append("Render")
        stripe_bg = st.radio("🖼️ Sfondo", bg_opts, horizontal=True,
            help="Master 1/2 = foto ferma. Calderone = foto in movimento. Render = glitch principale come sfondo, strisce sopra.")

        col_tog1, col_tog2 = st.columns(2)
        with col_tog1:
            stripe_glitch = st.toggle("⚡ Striscia glitchata", value=False,
                help="OFF = striscia pulita. ON = glitchata.")
        with col_tog2:
            stripe_reverse = st.toggle("🔄 Reverse", value=False,
                help="Inverte: strisce ferme, tutto il resto Calderone.")

        stripe_force_beat_react = st.toggle("🎵 Tutto a tempo", value=False,
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
                        help="Autonomo: la striscia scatta sui beat/onset rilevati anche senza 'Movimento random'. Se entrambi attivi, il movimento random accelera sul beat invece di scattare.")
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
                        help="Autonomo: la striscia scatta d'angolo sui beat/onset rilevati anche senza 'Rotazione automatica'. Se entrambe attive, la rotazione accelera sul beat invece di scattare.")

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
                        help="Autonomo: la lancetta scatta d'angolo sui beat/onset rilevati anche senza 'Rotazione automatica'. Se entrambe attive, la rotazione accelera sul beat invece di scattare.")

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
                        help="Autonomo: il cerchio scatta di raggio sui beat/onset rilevati anche senza 'Espansione ciclica'. Se entrambe attive, l'espansione accelera sul beat invece di scattare.")

                st.divider()

                # Effetti comuni
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    chroma_on = st.toggle("🌈 Chroma", value=False, key=f"ch_{i}")
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

        # --- ANTEPRIMA (renderizzata a sinistra, sotto "Carica audio") ---
        prev_choices = []
        prev_files   = {}
        if up_m1: prev_choices.append("Master 1");             prev_files["Master 1"] = up_m1
        if up_m2: prev_choices.append("Master 2");             prev_files["Master 2"] = up_m2
        if up_t:  prev_choices.append("Prima foto Calderone"); prev_files["Prima foto Calderone"] = up_t[0]

        with stripe_preview_slot:
         st.caption("🔍 Anteprima strisce")
         if prev_choices:
            prev_sel = st.selectbox("Anteprima su", prev_choices, label_visibility="collapsed")
            pf = prev_files[prev_sel]
            pf.seek(0)
            prev_img   = np.array(Image.open(pf).convert("RGB"))
            ph, pw     = prev_img.shape[:2]
            scale      = 160 / max(ph, pw)
            dw, dh     = int(pw * scale), int(ph * scale)
            prev_small = cv2.resize(prev_img, (dw, dh))
            overlay    = prev_small.copy()

            def _draw_stripe_h(ov, p0, p1, l0, l1):
                p0,p1 = max(0,p0), min(dh,p1)
                l0,l1 = max(0,l0), min(dw,l1)
                if p1>p0 and l1>l0:
                    ov[p0:p1, l0:l1] = (ov[p0:p1, l0:l1]*0.35 + np.array([120,80,220])*0.65).astype(np.uint8)
                if p0>1: ov[max(0,p0-2):p0, l0:l1] = [80,40,200]
                if p1<dh: ov[p1:min(dh,p1+2), l0:l1] = [80,40,200]

            def _draw_stripe_v(ov, p0, p1, l0, l1):
                p0,p1 = max(0,p0), min(dw,p1)
                l0,l1 = max(0,l0), min(dh,l1)
                if p1>p0 and l1>l0:
                    ov[l0:l1, p0:p1] = (ov[l0:l1, p0:p1]*0.35 + np.array([120,80,220])*0.65).astype(np.uint8)
                if p0>1: ov[l0:l1, max(0,p0-2):p0] = [80,40,200]
                if p1<dw: ov[l0:l1, p1:min(dw,p1+2)] = [80,40,200]

            for idx, s in enumerate(stripes):
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
                    corners_r = np.array([[-half_w_r,-half_h_r],[half_w_r,-half_h_r],
                                          [half_w_r,half_h_r],[-half_w_r,half_h_r]], dtype=np.float32)
                    rot_r = np.array([[cos_r,-sin_r],[sin_r,cos_r]])
                    pts_r = ((rot_r @ corners_r.T).T + np.array([cx_r,cy_r])).astype(np.int32)
                    mask_r = np.zeros((dh, dw), dtype=np.uint8)
                    cv2.fillPoly(mask_r, [pts_r], 255)
                    m3_r = mask_r[:,:,np.newaxis] / 255.0
                    overlay[:] = (overlay*(1-m3_r*0.65) + np.array([120,80,220])*m3_r*0.65).astype(np.uint8)
                    cv2.polylines(overlay, [pts_r], True, (80,40,200), 2)

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
                    radius = int(s.get("radius", 30) / 100 * max(dh, dw) / 2)
                    radius = max(1, radius)
                    if s.get("filled", True):
                        mask_c = np.zeros((dh, dw), dtype=np.uint8)
                        cv2.circle(mask_c, (cx, cy), radius, 255, -1)
                        m3 = mask_c[:, :, np.newaxis] / 255.0
                        overlay[:] = (overlay * (1 - m3 * 0.65) + VIOLET * m3 * 0.65).astype(np.uint8)
                    else:
                        cv2.circle(overlay, (cx, cy), radius, (120, 80, 220), max(2, s.get("size", 8)))


            caption = "Anteprima — viola = striscia attiva (lunghezza e centro come impostati)"
            if stripe_reverse:
                caption += " · REVERSE ON"
            st.image(overlay, caption=caption, use_container_width=True)
         else:
            st.caption("Carica almeno una foto per vedere l'anteprima.")

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
            if uploaded_preset:
                try:
                    loaded = json.load(uploaded_preset)
                    st.session_state.preset = loaded
                    # Ripristina gli ID strisce in base al numero salvato nel preset
                    n_loaded = len(loaded.get("stripes", []))
                    if n_loaded > 0:
                        st.session_state.stripe_ids = list(range(n_loaded))
                        st.session_state.stripe_next_id = n_loaded
                    st.success("Preset caricato — ricarica la pagina per applicarlo.")
                except Exception as e:
                    st.error(f"Errore preset: {e}")

    else:
        stripe_chroma = False
        stripe_flash  = False

    st.divider()

    # --- EFFETTI GLOBALI — disponibili sempre, anche senza strisce selettive ---
    st.caption("✨ Effetti globali sul render")
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
    fmt = st.selectbox("Format", ["16:9 (Orizzontale)", "9:16 (Verticale)", "1:1 (Quadrato)"])
    dur = st.number_input("Durata (sec)", 1, 300, 10)
    st.session_state.dur_value = int(dur)

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
            horizontal=True, help="Se il detect sbaglia (es. tracce con poca batteria), scrivi tu il BPM.")
        if bpm_mode == "Inserisci manualmente":
            manual_bpm = st.number_input("BPM", min_value=40, max_value=220, value=120, step=1)

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

            if preview_result['detected_bpm'] > 0:
                st.info(f"🎯 BPM {preview_result['bpm_source'].lower()}: **{preview_result['detected_bpm']:.1f}**")

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
            manual_bpm=manual_bpm, onset_sensitivity=onset_sensitivity
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
