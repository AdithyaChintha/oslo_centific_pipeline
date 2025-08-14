def fmt_hhmmss_ms(t: float) -> str:
    """Format seconds into HH:MM:SS.mmm"""
    ms = int(round((t - int(t)) * 1000))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"




