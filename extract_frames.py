import argparse
import os
import time

import cv2

# ================= MUST MATCH send.py EXACTLY =================
FRAME_PAYLOAD_BYTES = 2048
HEADER_BYTES = 8
# =================================================================

DATA_CAPACITY = FRAME_PAYLOAD_BYTES - HEADER_BYTES  # bytes actually available per frame

# Quality levels to try, high to low, before also shrinking resolution.
QUALITY_LADDER = [90, 80, 70, 60, 50, 40, 30, 20]
# If even the lowest quality doesn't fit, shrink resolution by this factor
# and retry the whole quality ladder again, up to this many times.
MAX_RESIZE_STEPS = 4
RESIZE_FACTOR = 0.8


def compress_to_fit(frame, max_bytes=DATA_CAPACITY):
    """
    Tries progressively lower jpg quality, then progressively smaller
    resolution, until the encoded frame fits under max_bytes. Returns
    (jpg_bytes, quality_used, resolution_used). Raises if even the
    smallest/lowest-quality attempt still doesn't fit.
    """
    working_frame = frame
    for _resize_step in range(MAX_RESIZE_STEPS + 1):
        for quality in QUALITY_LADDER:
            ok, encoded = cv2.imencode('.jpg', working_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                continue
            jpg_bytes = encoded.tobytes()
            if len(jpg_bytes) <= max_bytes:
                return jpg_bytes, quality, working_frame.shape[:2]
        # ran out of quality steps at this resolution -- shrink and try again
        h, w = working_frame.shape[:2]
        new_w, new_h = int(w * RESIZE_FACTOR), int(h * RESIZE_FACTOR)
        if new_w < 16 or new_h < 16:
            break
        working_frame = cv2.resize(working_frame, (new_w, new_h))

    raise RuntimeError(
        f"Could not compress a frame under {max_bytes} bytes even at the "
        f"lowest quality/resolution tried. Consider raising FRAME_PAYLOAD_BYTES "
        f"in both this script and send.py, or starting from a lower-resolution source."
    )


def extract_from_webcam(cap, args):
    saved_count = 0
    frame_interval = 1.0 / args.fps
    next_capture_time = 0.0
    t_start = time.time()

    while (time.time() - t_start) < args.duration:
        ret, frame = cap.read()
        if not ret:
            break
        elapsed = time.time() - t_start
        if elapsed >= next_capture_time:
            jpg_bytes, quality, shape = compress_to_fit(frame)
            out_path = os.path.join(args.out, f"frame_{saved_count:06d}.jpg")
            with open(out_path, 'wb') as f:
                f.write(jpg_bytes)
            saved_count += 1
            next_capture_time += frame_interval

    return saved_count


def extract_from_file(cap, args, source_fps):
    # Step through by source-fps ratio so --fps is honored regardless of
    # the file's own native frame rate.
    keep_every = max(1, round(source_fps / args.fps))
    max_frames = int(args.duration * source_fps) if args.duration is not None else None

    saved_count = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and idx >= max_frames:
            break
        if idx % keep_every == 0:
            jpg_bytes, quality, shape = compress_to_fit(frame)
            out_path = os.path.join(args.out, f"frame_{saved_count:06d}.jpg")
            with open(out_path, 'wb') as f:
                f.write(jpg_bytes)
            saved_count += 1
        idx += 1

    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Extract video/webcam frames into jpgs for send.py.")
    parser.add_argument('--source', default='0',
                         help="video file path, or a camera index like 0 for the default webcam")
    parser.add_argument('--fps', type=float, default=10.0,
                         help="frames per second to EXTRACT AT (independent of source video's own fps)")
    parser.add_argument('--duration', type=float, default=None,
                         help="seconds to capture (required for a live webcam; optional cap for a video file)")
    parser.add_argument('--out', default='frames',
                         help="folder to write frame_NNNNNN.jpg files into")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # allow --source to be either a file path or a webcam index typed as a number
    source = int(args.source) if args.source.isdigit() else args.source
    is_webcam = isinstance(source, int)

    if is_webcam and args.duration is None:
        raise ValueError("--duration is required when --source is a webcam index (there's no natural end point).")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source!r}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # some files/cameras report 0, fall back to 30

    print(f"Source: {args.source} ({'webcam' if is_webcam else 'file'}), reports {source_fps:.1f} fps")
    print(f"Extracting at {args.fps} fps -> writing frame_NNNNNN.jpg into '{args.out}/'")
    print(f"Each frame will be compressed to fit under {DATA_CAPACITY} bytes (matching send.py's frame budget)")

    if is_webcam:
        saved_count = extract_from_webcam(cap, args)
    else:
        saved_count = extract_from_file(cap, args, source_fps)

    cap.release()
    print(f"Done: wrote {saved_count} frame(s) to '{args.out}/'")


if __name__ == "__main__":
    main()
