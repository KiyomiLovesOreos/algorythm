"""
Example: Optimized MP3 Visualizer
  
Shows best practices for fast video rendering from MP3 files.
"""

from algorythm.audio_loader import visualize_audio_file
from algorythm.visualization import (
    CircularVisualizer,
    WaveformVisualizer,
    SpectrogramVisualizer
)


def quick_test_render():
    """Test render with just 10 seconds - perfect for testing settings."""
    print("\n" + "=" * 60)
    print("Quick Test Render (10 seconds)")
    print("=" * 60)
    print("\nThis renders only the first 10 seconds to test your settings.")
    print("Once you like the result, render the full song.\n")
    
    # Fast circular visualizer
    viz = CircularVisualizer(
        sample_rate=44100,
        num_bars=64,
        smoothing=0.7
    )
    
    visualize_audio_file(
        input_file='path/to/your/song.mp3',  # Change this!
        output_file='test_10sec.mp4',
        visualizer=viz,
        video_width=1280,   # 720p
        video_height=720,
        video_fps=24,
        duration=10.0       # Only 10 seconds!
    )
    
    print("\n✓ Check test_10sec.mp4 - if you like it, run full render!")


def optimized_full_render():
    """Optimized settings for full song (2-3 minutes)."""
    print("\n" + "=" * 60)
    print("Optimized Full Song Render")
    print("=" * 60)
    print("\nThis uses optimal settings for speed while maintaining quality.\n")
    
    # Circular visualizer - fast and looks great
    viz = CircularVisualizer(
        sample_rate=44100,
        num_bars=64,        # 64 bars is good balance
        inner_radius=0.2,
        smoothing=0.7
    )
    
    visualize_audio_file(
        input_file='path/to/your/song.mp3',  # Change this!
        output_file='full_song_optimized.mp4',
        visualizer=viz,
        video_width=1280,   # 720p (2-3x faster than 1080p)
        video_height=720,
        video_fps=24        # 24fps (20% faster than 30fps)
    )
    
    print("\n✓ Full song rendered with optimized settings!")


def ultra_fast_render():
    """Ultra-fast render for very long songs or slow computers."""
    print("\n" + "=" * 60)
    print("Ultra-Fast Render (480p)")
    print("=" * 60)
    print("\nUses lowest settings for maximum speed.")
    print("Good for: long songs (5+ min) or slower computers.\n")
    
    # Waveform is the fastest visualizer
    viz = WaveformVisualizer(
        sample_rate=44100,
        downsample_factor=2  # Extra speed boost
    )
    
    visualize_audio_file(
        input_file='path/to/your/song.mp3',  # Change this!
        output_file='ultra_fast.mp4',
        visualizer=viz,
        video_width=854,    # 480p (very fast!)
        video_height=480,
        video_fps=24
    )
    
    print("\n✓ Ultra-fast render complete!")


def high_quality_render():
    """High quality for special projects (slower but beautiful)."""
    print("\n" + "=" * 60)
    print("High Quality Render (1080p)")
    print("=" * 60)
    print("\nUses best settings for quality. Will take longer!")
    print("Recommended for: short songs (< 1 minute) or final exports.\n")
    
    # Circular with more bars for detail
    viz = CircularVisualizer(
        sample_rate=44100,
        num_bars=128,       # More bars = more detail
        inner_radius=0.15,
        smoothing=0.8
    )
    
    visualize_audio_file(
        input_file='path/to/your/song.mp3',  # Change this!
        output_file='high_quality.mp4',
        visualizer=viz,
        video_width=1920,   # Full HD
        video_height=1080,
        video_fps=30        # Smooth 30fps
    )
    
    print("\n✓ High quality render complete!")


def spectrogram_optimized():
    """Optimized spectrogram render (now much faster!)."""
    print("\n" + "=" * 60)
    print("Optimized Spectrogram Render")
    print("=" * 60)
    print("\nSpectrograms are now optimized with vectorized FFT.")
    print("Still slower than circular/waveform, but much better!\n")
    
    # Spectrogram with optimized settings
    viz = SpectrogramVisualizer(
        sample_rate=44100,
        window_size=2048,   # Good balance
        hop_size=512
    )
    
    visualize_audio_file(
        input_file='path/to/your/song.mp3',  # Change this!
        output_file='spectrogram_optimized.mp4',
        visualizer=viz,
        video_width=1280,   # 720p recommended
        video_height=720,
        video_fps=24
    )
    
    print("\n✓ Optimized spectrogram complete!")


def render_in_chunks():
    """Process long song in chunks if computer struggles."""
    print("\n" + "=" * 60)
    print("Chunk Rendering (for very long songs)")
    print("=" * 60)
    print("\nBreaks song into 30-second chunks.")
    print("Good for: songs > 5 minutes or limited RAM.\n")
    
    viz = CircularVisualizer(sample_rate=44100, num_bars=64)
    
    # Process first minute in 30-second chunks
    for i in range(2):  # 2 chunks = 1 minute
        offset = i * 30
        print(f"\nRendering chunk {i+1}/2 (seconds {offset}-{offset+30})...")
        
        visualize_audio_file(
            input_file='path/to/your/song.mp3',  # Change this!
            output_file=f'chunk_{i+1}.mp4',
            visualizer=viz,
            video_width=1280,
            video_height=720,
            video_fps=24,
            offset=offset,     # Start position
            duration=30.0      # 30 seconds
        )
    
    print("\n✓ All chunks rendered!")
    print("You can combine them later using ffmpeg or video editing software.")


def main():
    """Show all optimization examples."""
    print("\n" + "=" * 60)
    print("MP3 VISUALIZER - OPTIMIZED EXAMPLES")
    print("=" * 60)
    print("\nThese examples show how to render videos FAST!")
    print("\nIMPORTANT: Edit the file paths before running!")
    print("Replace 'path/to/your/song.mp3' with your actual MP3 file.\n")
    
    print("\nAvailable examples:")
    print("  1. quick_test_render()      - 10 second test (START HERE!)")
    print("  2. optimized_full_render()  - Full song, optimized")
    print("  3. ultra_fast_render()      - Ultra fast 480p")
    print("  4. high_quality_render()    - High quality 1080p")
    print("  5. spectrogram_optimized()  - Optimized spectrogram")
    print("  6. render_in_chunks()       - Chunk rendering")
    
    print("\n" + "=" * 60)
    print("Performance Tips:")
    print("=" * 60)
    print("✓ 720p is 2-3x faster than 1080p")
    print("✓ 24fps is 20% faster than 30fps")
    print("✓ Circular/Waveform are fastest visualizers")
    print("✓ Test with 10 seconds first!")
    print("✓ Close other apps to free RAM")
    print("\nFor more tips, see: PERFORMANCE_TIPS.md")
    print("=" * 60 + "\n")
    
    # Uncomment the example you want to run:
    # quick_test_render()
    # optimized_full_render()
    # ultra_fast_render()
    # high_quality_render()
    # spectrogram_optimized()
    # render_in_chunks()


if __name__ == '__main__':
    main()
