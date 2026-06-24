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


def kf_ui(param, label, min_v, max_v, default_v, step_v, stripe_id, dur, key_prefix):
    """
    Mostra UI per aggiungere/rimuovere keyframe di un parametro.
    Salva i KF in st.session_state[key_prefix].
    Ritorna la lista di KF corrente.
    """
    state_key = f"kf_{key_prefix}_{stripe_id}_{param}"
    if state_key not in st.session_state:
        st.session_state[state_key] = []

    kfs = st.session_state[state_key]

    # Pulsante fisarmonica
    toggle_key = f"kf_open_{key_prefix}_{stripe_id}_{param}"
    if toggle_key not in st.session_state:
        st.session_state[toggle_key] = False

    n_kf = len(kfs)
    btn_label = f"🎬 KF {label} ({n_kf})" if n_kf > 0 else f"🎬 KF {label}"
    if st.button(btn_label, key=f"kfbtn_{key_prefix}_{stripe_id}_{param}",
                 help=f"Aggiungi/gestisci keyframe per {label}"):
        st.session_state[toggle_key] = not st.session_state[toggle_key]

    if st.session_state[toggle_key]:
        with st.container(border=True):
            st.caption(f"⏱️ Keyframe: {label}")
            # Aggiungi nuovo KF
            c_t, c_v, c_add = st.columns([2, 3, 1])
            with c_t:
                new_t = st.slider(f"t (sec)##{key_prefix}_{stripe_id}_{param}",
                                  0.0, float(max(dur, 1)), 0.0, step=0.1,
                                  key=f"kf_nt_{key_prefix}_{stripe_id}_{param}")
            with c_v:
                new_v = st.slider(f"valore##{key_prefix}_{stripe_id}_{param}",
                                  float(min_v), float(max_v), float(default_v), step=float(step_v),
                                  key=f"kf_nv_{key_prefix}_{stripe_id}_{param}")
            with c_add:
                st.write("")
                if st.button("➕", key=f"kf_add_{key_prefix}_{stripe_id}_{param}",
                             help="Aggiungi keyframe"):
                    kfs.append({'t': round(new_t, 2), 'v': round(new_v, 4)})
                    kfs.sort(key=lambda k: k['t'])
                    st.session_state[state_key] = kfs
                    st.rerun()

            # Lista KF esistenti
            to_del = None
            for ki, kf in enumerate(kfs):
                ck1, ck2, ck3 = st.columns([2, 3, 1])
                with ck1:
                    st.caption(f"t = {kf['t']:.1f}s")
                with ck2:
                    st.caption(f"v = {kf['v']:.2f}")
                with ck3:
                    if st.button("🗑️", key=f"kf_del_{key_prefix}_{stripe_id}_{param}_{ki}"):
                        to_del = ki
            if to_del is not None:
                kfs.pop(to_del)
                st.session_state[state_key] = kfs
                st.rerun()

    return st.session_state[state_key]


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
        length_offset = s.get('offset_length', 50.0)
        if s.get('move_random', False):
            length_offset = offset
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
            angle = kf_get(s, 'angle', t, total_dur, s.get('angle', 90.0))
            if s.get('auto_rotate', False):
                angle = offset  # offset precomputato come angolo corrente
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
            if s.get('auto_expand', False):
                radius = offset  # offset precomputato come raggio crescente
            opacity = kf_get(s, 'opacity', t, total_dur, float(s.get('opacity', 1.0)))
            draw_cerchio(out, src_stripe, h, w,
                         s.get('cx', 50.0), s.get('cy', 50.0),
                         radius, s.get('filled', True),
                         int(kf_get(s, 'size', t, total_dur, s.get('size', 8))),
                         chroma_on, chroma_amt, mode, opacity)

        elif s_orient == "Striscia Ruotata":
            angle = kf_get(s, 'angle', t, total_dur, s.get('angle', 0.0))
            if s.get('auto_rotate', False):
                angle = offset
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
                    global_flash=False, global_flash_threshold=0.7, global_flash_intensity=100):

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

    # --- precomputo traiettorie per ogni striscia ---
    stripe_offsets_t = []
    if stripe_mode and stripes:
        rng = np.random.default_rng(42)
        t_arr = np.linspace(0, max_limit, total_f)
        for s in stripes:
            s_orient = s.get('orientation', stripe_orientation)

            if s_orient in ["Lancetta", "Striscia Ruotata"] and s.get('auto_rotate', False):
                # rotazione continua: angolo cresce linearmente nel tempo
                spd = s.get('rotate_speed', 30.0)  # gradi/secondo
                start_angle = s.get('angle', 90.0)
                traj = (start_angle + spd * t_arr) % 360.0
                stripe_offsets_t.append(traj)

            elif s_orient == "Cerchio" and s.get('auto_expand', False):
                # espansione ciclica: raggio cresce da 0 a 100 e riparte
                spd = s.get('expand_speed', 20.0)  # % al secondo
                traj = (spd * t_arr) % 100.0
                stripe_offsets_t.append(traj)

            elif s.get('move_random', False):
                spd = max(0.1, s.get('move_speed', 1.0))
                freq1 = spd * rng.uniform(0.1, 0.3)
                freq2 = spd * rng.uniform(0.05, 0.15)
                phase1, phase2 = rng.uniform(0, np.pi*2, 2)
                traj = (np.sin(2*np.pi*freq1*t_arr + phase1) * 0.5 +
                        np.sin(2*np.pi*freq2*t_arr + phase2) * 0.5)
                traj = (traj + 1) / 2 * 80 + 10
                stripe_offsets_t.append(traj)
            else:
                stripe_offsets_t.append(np.full(total_f, 50.0))

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

        st.divider()

        # --- Pulsanti aggiungi / gestione lista strisce ---
        col_add, col_ninfo = st.columns([1, 2])
        with col_add:
            if st.button("➕ Aggiungi striscia", key="add_stripe"):
                st.session_state.stripe_ids.append(st.session_state.stripe_next_id)
                st.session_state.stripe_next_id += 1
                st.rerun()
        with col_ninfo:
            st.caption(f"{len(st.session_state.stripe_ids)} striscia/e attive")

        # Raccogliamo quale striscia eliminare (se richiesto)
        _to_delete = None

        for _loop_idx, i in enumerate(list(st.session_state.stripe_ids)):
            col_title, col_del = st.columns([5, 1])
            with col_title:
                st.caption(f"Striscia {_loop_idx+1}")
            with col_del:
                if st.button("✕", key=f"del_stripe_{i}", help=f"Elimina striscia {_loop_idx+1}"):
                    _to_delete = i

            if _to_delete == i:
                continue  # saltiamo il render di questa striscia

            # orientamento individuale sempre disponibile
            orient_opts = ["Orizzontale", "Verticale", "Striscia Ruotata", "Lancetta", "Cerchio"]
            if stripe_orientation == "Mix H+V":
                orient_default = 0
            elif stripe_orientation in orient_opts:
                orient_default = orient_opts.index(stripe_orientation)
            else:
                orient_default = 0
            stripe_orient_i = st.radio(
                f"Forma striscia {i+1}", orient_opts,
                index=orient_default, horizontal=True, key=f"so_{i}")

            # --- controlli specifici per tipo ---
            s_dict = {
                'orientation':   stripe_orient_i,
                'chroma_amount': 6,
                'flash':         False,
                'blend_mode':    'Normal',
                'opacity':       1.0,
            }

            if stripe_orient_i in ["Orizzontale", "Verticale"]:
                ca, cb, cc = st.columns(3)
                with ca:
                    s_dict['center'] = st.slider(f"Centro {i+1} (%)", 0, 100, min(20+i*25,95), key=f"sc_{i}")
                with cb:
                    s_dict['size'] = st.slider(f"Spessore {i+1} (%)", 1, 100, 10, key=f"ss_{i}")
                with cc:
                    s_dict['length'] = float(st.slider(f"Lunghezza {i+1} (%)", 5, 100, 100, key=f"sl_{i}"))
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    s_dict['length_audio'] = st.toggle(f"Lunghezza reattiva {i+1}", value=False, key=f"la_{i}")
                with col_m2:
                    s_dict['move_random'] = st.toggle(f"Movimento random {i+1}", value=False, key=f"mr_{i}")
                s_dict['move_speed'] = 1.0
                if s_dict['move_random']:
                    s_dict['move_speed'] = st.slider(f"Velocità movimento {i+1}", 0.1, 5.0, 1.0, step=0.1, key=f"ms_{i}")
                s_dict['offset_length'] = float(st.slider(f"Offset dx/sx {i+1} (%)", 0, 100, 50, key=f"oc_{i}"))
                # --- KF ---
                kf_col1, kf_col2, kf_col3 = st.columns(3)
                with kf_col1:
                    s_dict.setdefault('keyframes', {})['center'] = kf_ui('center', 'Centro', 0, 100, s_dict['center'], 1, i, dur, 'hv')
                with kf_col2:
                    s_dict['keyframes']['size'] = kf_ui('size', 'Spessore', 1, 100, s_dict['size'], 1, i, dur, 'hv')
                with kf_col3:
                    s_dict['keyframes']['length'] = kf_ui('length', 'Lunghezza', 5, 100, s_dict['length'], 1, i, dur, 'hv')

            elif stripe_orient_i == "Striscia Ruotata":
                col_cx, col_cy = st.columns(2)
                with col_cx:
                    s_dict['cx'] = float(st.slider(f"Centro X {i+1} (%)", 0, 100, 50, key=f"rcx_{i}",
                        help="Centro orizzontale di rotazione"))
                with col_cy:
                    s_dict['cy'] = float(st.slider(f"Centro Y {i+1} (%)", 0, 100, 50, key=f"rcy_{i}",
                        help="Centro verticale di rotazione"))
                col_a, col_sp, col_l = st.columns(3)
                with col_a:
                    s_dict['angle'] = float(st.slider(f"Angolo {i+1} (°)", 0, 360, 0, key=f"rang_{i}",
                        help="0=orizzontale, 45=diagonale, 90=verticale, ecc."))
                with col_sp:
                    s_dict['size'] = float(st.slider(f"Spessore {i+1} (%)", 1, 100, 15, key=f"rsp_{i}",
                        help="Quanto è larga la striscia"))
                with col_l:
                    s_dict['length'] = float(st.slider(f"Lunghezza {i+1} (%)", 5, 150, 100, key=f"rl_{i}",
                        help="Quanto si estende (>100 = esce dai bordi)"))
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    s_dict['auto_rotate'] = st.toggle(f"Rotazione automatica {i+1}", value=False, key=f"rar_{i}")
                with col_r2:
                    s_dict['length_audio'] = st.toggle(f"Lunghezza reattiva {i+1}", value=False, key=f"la_{i}")
                s_dict['rotate_speed'] = 30.0
                if s_dict['auto_rotate']:
                    s_dict['rotate_speed'] = st.slider(f"Velocità rotazione {i+1} (°/sec)", 5.0, 360.0, 30.0, key=f"rrs_{i}")
                # --- KF ---
                kf_col1, kf_col2, kf_col3 = st.columns(3)
                with kf_col1:
                    s_dict.setdefault('keyframes', {})['angle'] = kf_ui('angle', 'Angolo', 0, 360, s_dict['angle'], 1, i, dur, 'rot')
                with kf_col2:
                    s_dict['keyframes']['size'] = kf_ui('size', 'Spessore', 1, 100, s_dict['size'], 1, i, dur, 'rot')
                with kf_col3:
                    s_dict['keyframes']['length'] = kf_ui('length', 'Lunghezza', 5, 150, s_dict['length'], 1, i, dur, 'rot')

            elif stripe_orient_i == "Lancetta":
                col_cx, col_cy = st.columns(2)
                with col_cx:
                    s_dict['cx'] = float(st.slider(f"Centro X {i+1} (%)", 0, 100, 50, key=f"lcx_{i}",
                        help="Punto di pivot orizzontale"))
                with col_cy:
                    s_dict['cy'] = float(st.slider(f"Centro Y {i+1} (%)", 0, 100, 50, key=f"lcy_{i}",
                        help="Punto di pivot verticale"))
                col_a, col_l, col_t = st.columns(3)
                with col_a:
                    s_dict['angle'] = float(st.slider(f"Angolo {i+1} (°)", 0, 360, 90, key=f"lang_{i}",
                        help="0=destra, 90=su, 180=sinistra, 270=giù"))
                with col_l:
                    s_dict['length'] = float(st.slider(f"Lunghezza {i+1} (%)", 5, 100, 50, key=f"ll_{i}"))
                with col_t:
                    s_dict['size'] = st.slider(f"Spessore {i+1} (px)", 2, 100, 15, key=f"lt_{i}")
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    s_dict['auto_rotate'] = st.toggle(f"Rotazione automatica {i+1}", value=False, key=f"lar_{i}")
                with col_r2:
                    s_dict['length_audio'] = st.toggle(f"Lunghezza reattiva {i+1}", value=False, key=f"la_{i}")
                s_dict['rotate_speed'] = 30.0
                if s_dict['auto_rotate']:
                    s_dict['rotate_speed'] = st.slider(f"Velocità rotazione {i+1} (°/sec)", 5.0, 360.0, 30.0, key=f"lrs_{i}")
                # --- KF ---
                kf_col1, kf_col2, kf_col3 = st.columns(3)
                with kf_col1:
                    s_dict.setdefault('keyframes', {})['angle'] = kf_ui('angle', 'Angolo', 0, 360, s_dict['angle'], 1, i, dur, 'lan')
                with kf_col2:
                    s_dict['keyframes']['length'] = kf_ui('length', 'Lunghezza', 5, 100, s_dict['length'], 1, i, dur, 'lan')
                with kf_col3:
                    s_dict['keyframes']['size'] = kf_ui('size', 'Spessore', 2, 100, s_dict['size'], 1, i, dur, 'lan')

            elif stripe_orient_i == "Cerchio":
                col_cx, col_cy = st.columns(2)
                with col_cx:
                    s_dict['cx'] = float(st.slider(f"Centro X {i+1} (%)", 0, 100, 50, key=f"ccx_{i}"))
                with col_cy:
                    s_dict['cy'] = float(st.slider(f"Centro Y {i+1} (%)", 0, 100, 50, key=f"ccy_{i}"))
                col_r, col_t = st.columns(2)
                with col_r:
                    s_dict['radius'] = float(st.slider(f"Raggio {i+1} (%)", 1, 100, 30, key=f"cr_{i}"))
                with col_t:
                    s_dict['size'] = st.slider(f"Spessore bordo {i+1} (px)", 1, 50, 8, key=f"ct_{i}",
                        help="Usato solo se Cerchio pieno è OFF")
                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    s_dict['filled'] = st.toggle(f"Cerchio pieno {i+1}", value=True, key=f"cf_{i}")
                with col_c2:
                    s_dict['length_audio'] = st.toggle(f"Raggio reattivo {i+1}", value=False, key=f"la_{i}",
                        help="Il raggio pulsa col volume")
                with col_c3:
                    s_dict['auto_expand'] = st.toggle(f"Espansione ciclica {i+1}", value=False, key=f"ce_{i}",
                        help="Il cerchio cresce e riparte dal centro")
                s_dict['expand_speed'] = 20.0
                if s_dict.get('auto_expand'):
                    s_dict['expand_speed'] = st.slider(f"Velocità espansione {i+1} (%/sec)", 5.0, 100.0, 20.0, key=f"ces_{i}")
                # --- KF ---
                kf_col1, kf_col2 = st.columns(2)
                with kf_col1:
                    s_dict.setdefault('keyframes', {})['radius'] = kf_ui('radius', 'Raggio', 1, 100, s_dict['radius'], 1, i, dur, 'cer')
                with kf_col2:
                    s_dict['keyframes']['size'] = kf_ui('size', 'Spessore bordo', 1, 50, s_dict['size'], 1, i, dur, 'cer')

            # --- effetti comuni a tutti i tipi ---
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                chroma_on = st.toggle(f"Chroma {i+1}", value=False, key=f"ch_{i}")
            with col_e2:
                s_dict['flash'] = st.toggle(f"Flash beat {i+1}", value=False, key=f"fl_{i}")
            if chroma_on:
                s_dict['chroma_amount'] = st.slider(f"Intensità chroma {i+1}", 1, 30, 6, key=f"ca_{i}")

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                s_dict['blend_mode'] = st.selectbox(f"Blend mode {i+1}",
                    ["Normal", "Screen", "Multiply", "Difference"], key=f"bm_{i}")
            with col_b2:
                s_dict['opacity'] = st.slider(f"Opacità {i+1} (%)", 0, 100, 100, key=f"op_{i}") / 100.0

            # --- KF opacità (comune a tutti i tipi) ---
            s_dict.setdefault('keyframes', {})['opacity'] = kf_ui(
                'opacity', 'Opacità', 0.0, 1.0, s_dict['opacity'], 0.01, i, dur, 'com')

            stripes.append(s_dict)

        # Applica l'eliminazione dopo aver renderizzato tutto
        if _to_delete is not None:
            st.session_state.stripe_ids.remove(_to_delete)
            st.rerun()

        st.divider()

        # --- ANTEPRIMA ---
        prev_choices = []
        prev_files   = {}
        if up_m1: prev_choices.append("Master 1");             prev_files["Master 1"] = up_m1
        if up_m2: prev_choices.append("Master 2");             prev_files["Master 2"] = up_m2
        if up_t:  prev_choices.append("Prima foto Calderone"); prev_files["Prima foto Calderone"] = up_t[0]

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
            st.image(overlay, caption=caption, use_container_width=False)
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
            stripe_bg, stripe_glitch, stripe_reverse,
            stripe_chroma, stripe_flash,
            global_chroma, global_chroma_amt,
            global_flash, global_flash_threshold, global_flash_intensity
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
