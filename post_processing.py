import numpy as np


def apply_gap_fill(preds, max_gap=15, background_class=0):
    """
    Fill short background gaps between identical behaviors.
    If background appears for <= max_gap frames between the same behavior on both sides,
    replace the background with that behavior.
    """
    preds = np.array(preds, dtype=int)
    i = 0
    while i < len(preds):
        if preds[i] != background_class:
            i += 1
            continue
        j = i
        while j < len(preds) and preds[j] == background_class:
            j += 1
        gap_length = j - i
        if gap_length <= max_gap and i > 0 and j < len(preds):
            before = preds[i - 1]
            after  = preds[j]
            if before == after and before != background_class:
                preds[i:j] = before
        i = j
    return preds


def apply_min_duration_filter(preds, min_duration=15, background_class=0):
    """
    Remove short predicted runs of non-background classes.
    Any contiguous run shorter than min_duration frames is replaced by
    the preceding class (or background if at the start).
    When a replacement is made, the scan restarts from i so that the newly
    patched-in class is also checked against min_duration.
    """
    preds = np.array(preds, dtype=int)
    i = 0
    while i < len(preds):
        cls = preds[i]
        if cls == background_class:
            i += 1
            continue
        j = i
        while j < len(preds) and preds[j] == cls:
            j += 1
        run_length = j - i
        if run_length < min_duration:
            replacement = preds[i - 1] if i > 0 else background_class
            preds[i:j] = replacement
            if replacement == cls:
                i = j
        else:
            i = j
    return preds


def apply_supported_rearing_merge(preds, supported_class, unsupported_class, max_trailing_frames=30):
    """
    Reclassify unsupportedrearing frames that immediately follow a supportedrearing
    run as supportedrearing — the mouse is just descending from the wall.
    Only merges trailing unsupported runs up to max_trailing_frames long.
    """
    preds = np.array(preds, dtype=int)
    i = 0
    while i < len(preds):
        if preds[i] != unsupported_class:
            i += 1
            continue
        j = i
        while j < len(preds) and preds[j] == unsupported_class:
            j += 1
        if i > 0 and preds[i - 1] == supported_class and (j - i) <= max_trailing_frames:
            preds[i:j] = supported_class
        i = j
    return preds

