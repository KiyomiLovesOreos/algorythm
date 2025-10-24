#!/usr/bin/env python3
"""
Quick Start: Optimized MP3 Visualizer
Just edit the file path and run!
"""

from algorythm.audio_loader import visualize_audio_file
from algorythm.visualization import CircularVisualizer

# ============================================================
# EDIT THIS: Change to your MP3 file path
# ============================================================
INPUT_MP3 = 'path/to/your/song.mp3'  # <-- CHANGE THIS!
OUTPUT_MP4 = 'output_visualized.mp4'

# ============================================================
# RECOMMENDED SETTINGS (Fast + Good Quality)
# ============================================================
# These settings give best speed/quality balance for 2-3 min songs

viz = CircularVisualizer(
    sample_rate=44100,
    num_bars=64,
    smoothing=0.7
)

visualize_audio_file(
    input_file=INPUT_MP3,
    output_file=OUTPUT_MP4,
    visualizer=viz,
    video_width=1280,   # 720p - 2-3x faster than 1080p
    video_height=720,
    video_fps=24        # 24fps - 20% faster than 30fps
)

# ============================================================
# OTHER PRESET OPTIONS (Uncomment to use)
# ============================================================

# ULTRA FAST (for slow computers or long songs)
"""
visualize_audio_file(
    INPUT_MP3, 'ultra_fast.mp4', viz,
    video_width=854, video_height=480, video_fps=24
)
"""

# HIGH QUALITY (for short songs < 1 min)
"""
visualize_audio_file(
    INPUT_MP3, 'high_quality.mp4', viz,
    video_width=1920, video_height=1080, video_fps=30
)
"""

# TEST FIRST 10 SECONDS (always do this first!)
"""
visualize_audio_file(
    INPUT_MP3, 'test_10sec.mp4', viz,
    video_width=1280, video_height=720, video_fps=24,
    duration=10.0  # Only 10 seconds!
)
"""

print("\n✓ Done! Check:", OUTPUT_MP4)
