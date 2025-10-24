# 🎬 Video Export for Beginners

## What is Video Export?

Instead of making just sound files (like `.wav` or `.mp3`), you can now make **music videos** with cool visualizations! The music syncs perfectly with animated visuals.

---

## 🚀 Super Simple Start

### 1. Install What You Need

```bash
# Install the video tools
pip install opencv-python

# On Linux/Mac, also install ffmpeg:
# Ubuntu: sudo apt-get install ffmpeg
# Mac: brew install ffmpeg
```

### 2. Make Your First Video

Copy this code - it's exactly like making audio, but with `video=True`:

```python
from algorythm.synth import SynthPresets
from algorythm.sequence import Motif, Scale
from algorythm.structure import Composition

# Make your song (same as always!)
song = Composition(tempo=120)
song.add_track('Lead', SynthPresets.pluck())
song.repeat_motif(Motif.from_intervals([0, 2, 4, 7], scale=Scale.major('C')), bars=4)

# The magic line - just add video=True!
song.render('my_first_video.mp4', video=True)
```

That's it! You now have a music video! 🎉

---

## 🎨 5 Types of Visualizations

### 1. **Spectrum** (Default) - Bouncing Bars
Like an equalizer with bars that bounce to the beat!

```python
song.render('spectrum.mp4', video=True, video_config={
    'visualizer': 'spectrum'
})
```

**Try this for:** Any music, especially electronic or bass-heavy

---

### 2. **Waveform** - Sound Waves
Shows the actual sound wave moving across the screen!

```python
song.render('waveform.mp4', video=True, video_config={
    'visualizer': 'waveform'
})
```

**Try this for:** Calm music, vocals, acoustic songs

---

### 3. **Circular** - Round Equalizer
Bars in a circle that pulse outward - looks super cool!

```python
song.render('circular.mp4', video=True, video_config={
    'visualizer': 'circular'
})
```

**Try this for:** Dance music, upbeat songs, when you want something fancy

---

### 4. **Particle** - Flying Dots
Animated particles that fly around and react to the music!

```python
song.render('particles.mp4', video=True, video_config={
    'visualizer': 'particle'
})
```

**Try this for:** Ambient music, chill songs, artistic videos

---

### 5. **Spectrogram** - Frequency Heatmap
Shows all the frequencies over time in a colorful map!

```python
song.render('spectrogram.mp4', video=True, video_config={
    'visualizer': 'spectrogram'
})
```

**Try this for:** Complex music, when you want to show detail

---

## 🌈 Change the Colors!

Make your videos match your style:

```python
song.render('colorful.mp4', video=True, video_config={
    'visualizer': 'circular',
    'background_color': (0, 0, 50),        # Dark blue background
    'foreground_color': (255, 100, 200)    # Pink foreground
})
```

### Easy Color Combos to Try:

```python
# Neon Cyan
'background_color': (0, 0, 0), 'foreground_color': (0, 255, 255)

# Hot Pink
'background_color': (20, 0, 20), 'foreground_color': (255, 0, 200)

# Electric Green
'background_color': (0, 0, 0), 'foreground_color': (0, 255, 100)

# Golden
'background_color': (0, 0, 0), 'foreground_color': (255, 215, 0)

# Purple Dream
'background_color': (10, 0, 30), 'foreground_color': (200, 100, 255)

# Fire Orange
'background_color': (10, 0, 0), 'foreground_color': (255, 150, 0)
```

**Color Tip:** Use RGB values (0-255 for each: Red, Green, Blue)

---

## 📱 Make Videos for Social Media

### YouTube Video (Full HD)
```python
song.render('youtube.mp4', video=True, video_config={
    'width': 1920,
    'height': 1080
})
```

### Instagram Post (Square)
```python
song.render('instagram.mp4', video=True, video_config={
    'width': 1080,
    'height': 1080
})
```

### TikTok/Stories (Vertical)
```python
song.render('tiktok.mp4', video=True, video_config={
    'width': 1080,
    'height': 1920
})
```

---

## 🎮 Fun Challenges

### Challenge 1: Rainbow Bars
Make a spectrum visualizer with rainbow colors!
```python
song.render('rainbow.mp4', video=True, video_config={
    'visualizer': 'spectrum',
    'foreground_color': (255, 0, 255)  # Try different colors!
})
```

### Challenge 2: Spinning Circle
Make a circular visualizer with lots of bars:
```python
song.render('spinner.mp4', video=True, video_config={
    'visualizer': 'circular',
    'num_bars': 128,  # More bars = cooler effect!
    'foreground_color': (0, 255, 255)
})
```

### Challenge 3: Particle Storm
Make tons of particles react to your music:
```python
song.render('storm.mp4', video=True, video_config={
    'visualizer': 'particle',
    'num_particles': 300,  # More particles!
    'sensitivity': 2.0      # React more to the music!
})
```

---

## 💡 Quick Tips

1. **Start Simple**: First try with just `video=True`, then add colors
2. **Experiment**: Try all 5 visualizers to see which you like best
3. **Match the Mood**: 
   - Fast song = Spectrum or Circular
   - Slow song = Waveform or Particles
4. **File Location**: Videos save to `~/Music/` folder by default
5. **Be Patient**: Videos take longer to create than audio (that's normal!)

---

## 🎯 Complete Example: Make a Cool Video

```python
from algorythm.synth import SynthPresets
from algorythm.sequence import Motif, Scale
from algorythm.structure import Composition, Reverb

print("🎬 Making an awesome music video...")

# Create your song
song = Composition(tempo=140)

# Add a cool lead
lead = SynthPresets.bell()
melody = Motif.from_intervals([0, 2, 4, 7, 9, 7, 4, 2], scale=Scale.major('D'))
song.add_track('Lead', lead).repeat_motif(melody, bars=8)

# Add some bass
bass = SynthPresets.bass()
bass_line = Motif.from_intervals([0, 0, 0, 0], scale=Scale.major('D'))
song.add_track('Bass', bass).repeat_motif(bass_line, bars=8)

# Add effects
song.add_fx(Reverb(mix=0.4))

# Export as a rad video!
song.render('my_awesome_video.mp4', video=True, video_config={
    'visualizer': 'circular',
    'width': 1920,
    'height': 1080,
    'background_color': (0, 0, 0),
    'foreground_color': (0, 255, 200),  # Turquoise!
    'num_bars': 80,
    'smoothing': 0.7
})

print("✨ Done! Check ~/Music/my_awesome_video.mp4")
```

---

## 🆘 Help! Something's Not Working

### "opencv-python not found"
```bash
pip install opencv-python
```

### "ffmpeg not found"
You need to install ffmpeg on your computer:
- **Ubuntu**: `sudo apt-get install ffmpeg`
- **Mac**: `brew install ffmpeg`
- **Windows**: Download from ffmpeg.org

### Video has no sound
Make sure ffmpeg is installed! Test with: `ffmpeg -version`

### It's taking forever!
Videos take time to make, especially:
- Longer songs = longer wait
- Higher quality (1080p) = longer wait
- More particles = longer wait

Try smaller settings first (720p, fewer particles), then go bigger!

---

## 🎉 You're a Video Creator Now!

You can now make:
- ✅ Music videos for YouTube
- ✅ Posts for Instagram
- ✅ Content for TikTok
- ✅ Cool visualizations for any platform!

**Remember:** Start simple, experiment often, and have fun! 🎵✨🎬
