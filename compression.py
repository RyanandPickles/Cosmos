"""
compression.py
---------------
Video-frame compression for the Cosmos SDR video link.

WHY JPEG (MJPEG) INSTEAD OF A "REAL" VIDEO CODEC (H.264/H.265)
----------------------------------------------------------------
A true video codec gets its extra compression mostly from *inter-frame*
prediction (motion compensation): frame N is described as a delta from
frame N-1. That's great over a clean channel, but this link is a raw
QAM/LDPC PHY over the air -- a burst of bit errors or a dropped frame
will corrupt the reference frame, and every subsequent P/B-frame that
depends on it decodes into garbage until the next keyframe. Recovering
from that means periodic keyframes (I-frames), GOP buffering, and a lot
of extra latency/complexity that fights the "continuous 30 fps,
low-latency" goal.

MJPEG-style compression (each frame compressed *independently*) gives
up the inter-frame gains but buys back something more valuable for this
link: every frame is a clean, self-contained recovery point. One bad
frame never contaminates the next. That matches how real analog/digital
FPV video links and many robotics video links actually work. It also
lets us hit an exact, fixed, per-frame bit budget every time (see
below), which the fixed-length OFDM-less frame structure in cosmos.py
needs (the receiver must know num_transmit_symbols ahead of time).

If/when a cleaner, higher-throughput channel is available, swap this
module's encode_frame/decode_frame for an H.264 encoder (e.g. via
`cv2.VideoWriter` with a hardware/software H.264 fourcc, or shelling
out to ffmpeg) without touching anything downstream -- transport.py
only cares about "bytes in, bytes out".

WHAT THIS MODULE DOES
----------------------
- Resizes + JPEG-encodes a webcam frame.
- Adaptively searches JPEG quality (and, if needed, resolution) so the
  compressed frame fits under a caller-specified byte budget. This is
  what makes constant, predictable-size frames possible over a
  fixed-symbol-count radio link.
- Decodes bytes back into an image on the receive side.
- Small numpy <-> bit-string helpers so the output plugs directly into
  ldpc.py (which speaks in '0'/'1' strings).
"""

import numpy as np
import cv2


# ---------------------------------------------------------------
# Frame <-> compressed bytes
# ---------------------------------------------------------------
def resize_frame(frame, width, height):
    """Resize a BGR frame to (width, height). Downscaling is the single
    biggest lever for fitting a video into a tiny RF bit budget -- cutting
    each dimension in half cuts raw pixel count (and roughly JPEG size) by 4x."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def encode_frame(frame, quality):
    """
    JPEG-encode a BGR frame (as produced by cv2.VideoCapture) at a given
    quality (0-100, higher = better/bigger).

    Returns: bytes (the compressed JPEG file contents)
    """
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed.")
    return buf.tobytes()


def decode_frame(jpg_bytes):
    """
    Decode JPEG bytes back into a BGR image (np.ndarray, HxWx3, uint8).
    Returns None if the bytes are too corrupted to decode (e.g. LDPC
    could not fully correct the block) -- callers should handle that by
    reusing/holding the previous good frame instead of crashing.
    """
    if jpg_bytes is None or len(jpg_bytes) == 0:
        return None
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame  # None on failure -- cv2 already handles that gracefully


# ---------------------------------------------------------------
# Adaptive compression to a fixed byte budget
# ---------------------------------------------------------------
def compress_to_budget(frame, max_bytes, width=None, height=None,
                        min_quality=5, max_quality=90, max_downscale_steps=4):
    """
    Compress `frame` so the resulting JPEG is <= max_bytes.

    Strategy:
      1. Optionally resize to (width, height) first (your chosen
         transmit resolution -- pick this once for the whole stream).
      2. Binary-search JPEG quality in [min_quality, max_quality] for the
         highest quality that still fits in max_bytes.
      3. If even min_quality doesn't fit, progressively halve the
         resolution (up to max_downscale_steps times) and retry -- some
         budget is simply too small for a given resolution, especially
         for a busy/high-detail frame.

    Returns
    -------
    jpg_bytes : bytes, guaranteed len(jpg_bytes) <= max_bytes (unless the
                frame cannot be compressed under max_bytes even at 1x1 --
                practically never happens for sane budgets)
    quality   : int, JPEG quality actually used
    size_used : (width, height) actually used after any downscaling
    """
    if width is not None and height is not None:
        frame = resize_frame(frame, width, height)

    for _ in range(max_downscale_steps + 1):
        lo, hi = min_quality, max_quality
        best_jpg, best_q = None, None

        # Binary search for the highest quality that fits the budget.
        while lo <= hi:
            mid = (lo + hi) // 2
            jpg = encode_frame(frame, mid)
            if len(jpg) <= max_bytes:
                best_jpg, best_q = jpg, mid
                lo = mid + 1  # try higher quality
            else:
                hi = mid - 1  # too big, reduce quality

        if best_jpg is not None:
            h, w = frame.shape[:2]
            return best_jpg, best_q, (w, h)

        # Even minimum quality didn't fit -- shrink resolution and retry.
        h, w = frame.shape[:2]
        frame = resize_frame(frame, max(2, w // 2), max(2, h // 2))

    # Last resort: return the smallest thing we can produce, even if it
    # slightly exceeds max_bytes. Caller/transport layer should choose a
    # sane max_bytes so this branch is never actually hit.
    jpg = encode_frame(frame, min_quality)
    h, w = frame.shape[:2]
    return jpg, min_quality, (w, h)


class AdaptiveFrameCompressor:
    """
    Stateful wrapper for real-time use in send.py.

    Keeps a running estimate of the JPEG quality that hits the target
    budget, and starts each new frame's binary search near that estimate
    instead of from scratch -- video content is usually similar
    frame-to-frame, so this converges in 1-2 tries most of the time and
    keeps per-frame compression fast enough for 30 fps.
    """

    def __init__(self, max_bytes, width=None, height=None,
                 min_quality=5, max_quality=90):
        self.max_bytes = max_bytes
        self.width = width
        self.height = height
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.last_quality = max_quality  # optimistic first guess

    def compress(self, frame):
        if self.width is not None and self.height is not None:
            frame = resize_frame(frame, self.width, self.height)

        # Fast path: try last frame's quality first.
        jpg = encode_frame(frame, self.last_quality)
        if len(jpg) <= self.max_bytes:
            # Try nudging quality up a bit to use the available budget
            # more fully (cheap: at most a couple of extra encodes).
            q = self.last_quality
            while q < self.max_quality:
                q_next = min(self.max_quality, q + 5)
                jpg_next = encode_frame(frame, q_next)
                if len(jpg_next) <= self.max_bytes:
                    jpg, q = jpg_next, q_next
                else:
                    break
            self.last_quality = q
            return jpg, q, (frame.shape[1], frame.shape[0])

        # Slow path: full adaptive search (also handles downscaling).
        jpg, q, size_used = compress_to_budget(
            frame, self.max_bytes, width=None, height=None,
            min_quality=self.min_quality, max_quality=self.last_quality,
        )
        self.last_quality = q
        return jpg, q, size_used


# ---------------------------------------------------------------
# Bytes <-> bit-string helpers (to feed ldpc.py, which speaks '0'/'1' strings)
# ---------------------------------------------------------------
def bytes_to_bits(data):
    """bytes -> str of '0'/'1', MSB first per byte. Vectorized with
    np.unpackbits instead of a per-byte format()/join loop."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)  # MSB-first by default, matches format(byte,'08b')
    return (bits + ord('0')).tobytes().decode('ascii')


def bits_to_bytes(bit_string):
    """str of '0'/'1' -> bytes. Length must be a multiple of 8 (pad first).
    Vectorized with np.packbits instead of a per-8-chars int()/bytes loop."""
    n = len(bit_string) - (len(bit_string) % 8)
    bits = np.frombuffer(bit_string[:n].encode('ascii'), dtype=np.uint8) - ord('0')
    return np.packbits(bits).tobytes()


if __name__ == "__main__":
    # Local smoke test with a synthetic "frame" (no camera needed).
    rng = np.random.default_rng(0)
    fake_frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)

    budget = 4000  # bytes
    jpg, q, size_used = compress_to_budget(fake_frame, budget, width=320, height=240)
    print(f"Compressed to {len(jpg)} bytes (budget {budget}), quality={q}, size={size_used}")
    assert len(jpg) <= budget, "compression exceeded requested budget"

    bits = bytes_to_bits(jpg)
    recovered_bytes = bits_to_bytes(bits)
    assert recovered_bytes == jpg, "bytes<->bits round trip failed"

    decoded = decode_frame(jpg)
    print(f"Decoded frame shape: {decoded.shape}")
    print("compression.py self-test passed.")
