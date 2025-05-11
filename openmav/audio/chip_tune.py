"""
Implements violin and piano sound generation for token feedback.
"""
import numpy as np

class ChipTunePlayer:
    """Generates and plays violin and piano sounds for token generation feedback."""
    
    def __init__(self):
        """Initialize the pygame mixer for audio playback."""
        self.initialized = False
        try:
            import pygame
            from scipy.signal import butter, lfilter
            pygame.mixer.pre_init(frequency=44100, size=-16, channels=1)
            pygame.mixer.init()
            self.pygame = pygame
            self.butter = butter
            self.lfilter = lfilter
            self.initialized = True
            
            # Store last instrument used to alternate
            self.last_instrument = "piano"
        except ImportError:
            print("Warning: pygame or scipy not available. Audio feedback disabled.")
            self.initialized = False
    
    def play_chip_tone(self, token_id: int, entropy: float, duration: float = 0.5):
        """
        Play either a violin or piano sound based on token ID and entropy.
        
        Args:
            token_id:      The token ID to map to pitch
            entropy:       Attention entropy affects timbre and expression (0–∞)
            duration:      Length of the sound in seconds
        """
        if not self.initialized:
            return
        
        # Alternate between violin and piano
        self.last_instrument = "violin" if self.last_instrument == "piano" else "piano"
        
        # Parameters
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # Map token_id to a musical pitch on a scale
        base = 220.0  # A3
        major_scale = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals in semitones
        scale_degree = token_id % 7  # Choose one of 7 notes in the scale
        octave_offset = (token_id // 7) % 3 - 1  # -1, 0, or 1 octave offset
        
        # Calculate the frequency for the chosen note
        semitones = major_scale[scale_degree] + (12 * octave_offset)
        freq = base * (2 ** (semitones / 12))
        
        # Normalize entropy for sound parameters
        e_norm = min(max(entropy / 10, 0.0), 1.0)
        
        if self.last_instrument == "violin":
            wave = self._generate_violin(freq, t, e_norm)
        else:
            wave = self._generate_piano(freq, t, e_norm)
            
        # Volume affected by entropy
        vol = 0.5 + e_norm * 0.5  # 0.5-1.0 range
        
        # Convert to 16-bit PCM and play
        pcm = (wave * 32767).astype(np.int16)
        snd = self.pygame.sndarray.make_sound(pcm)
        snd.set_volume(vol)
        snd.play()
    
    def _generate_violin(self, freq, t, intensity):
        """Generate a violin-like sound."""
        # Harmonics - strong odd harmonics with varying amplitudes
        harmonics = [
            1.0 * np.sin(2 * np.pi * freq * t),                    # Fundamental
            0.9 * np.sin(2 * np.pi * freq * 2 * t),               # 2nd harmonic
            0.8 * np.sin(2 * np.pi * freq * 3 * t),               # 3rd harmonic
            0.6 * np.sin(2 * np.pi * freq * 4 * t),               # 4th harmonic
            0.4 * np.sin(2 * np.pi * freq * 5 * t),               # 5th harmonic
            0.2 * np.sin(2 * np.pi * freq * 6 * t)                # 6th harmonic
        ]
        
        # Combine harmonics
        wave = sum(harmonics)
        
        # ADSR envelope for violin (slow attack, steady sustain, slow release)
        attack_time = 0.1 + (1 - intensity) * 0.15  # Less intense = longer attack
        decay_time = 0.1
        sustain_level = 0.7
        release_time = 0.2
        
        # Time points for ADSR envelope
        t_attack = min(attack_time, len(t)/44100)
        t_decay = min(t_attack + decay_time, len(t)/44100)
        t_release = max(0, len(t)/44100 - release_time)
        
        # Create ADSR envelope
        envelope = np.zeros_like(t)
        attack_mask = t < t_attack
        decay_mask = (t >= t_attack) & (t < t_decay)
        sustain_mask = (t >= t_decay) & (t < t_release)
        release_mask = t >= t_release
        
        envelope[attack_mask] = t[attack_mask] / t_attack
        envelope[decay_mask] = 1 - (1 - sustain_level) * (t[decay_mask] - t_attack) / (t_decay - t_attack)
        envelope[sustain_mask] = sustain_level
        envelope[release_mask] = sustain_level * (1 - (t[release_mask] - t_release) / (len(t)/44100 - t_release))
        
        # Apply envelope
        wave = wave * envelope
        
        # Vibrato - rate and depth affected by intensity
        vibrato_rate = 5 + intensity * 3  # 5-8 Hz
        vibrato_depth = 0.005 + intensity * 0.01  # 0.005-0.015
        vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
        
        # Apply vibrato by modulating the playback "time"
        t_vibrato = t + vibrato
        
        # This is a simplified implementation that doesn't truly model time modulation
        # For a real implementation, we would need to resample the waveform
        # Instead, add a frequency modulation approximation
        wave += 0.1 * np.sin(2 * np.pi * freq * (1 + vibrato) * t)
        
        # Add a bit of noise for bow friction sound
        bow_noise = np.random.randn(len(t)) * 0.02 * intensity
        wave += bow_noise
        
        # Gentle filter to shape tone
        b, a = self.butter(2, 7000/(44100/2), btype='low')
        wave = self.lfilter(b, a, wave)
        
        return np.tanh(wave)  # Soft limiting
    
    def _generate_piano(self, freq, t, intensity):
        """Generate a piano-like sound."""
        # Multiple strings with slight detuning for realistic piano sound
        detuning = 0.0003  # Very slight detuning factor
        
        strings = [
            1.0 * np.sin(2 * np.pi * freq * t),                     # Main string
            0.9 * np.sin(2 * np.pi * freq * (1 - detuning) * t),    # Slightly flat string
            0.9 * np.sin(2 * np.pi * freq * (1 + detuning) * t)     # Slightly sharp string
        ]
        
        # Rich harmonics structure for piano
        harmonics = [
            1.0 * np.sin(2 * np.pi * freq * t),                     # Fundamental
            0.6 * np.sin(2 * np.pi * freq * 2 * t),                # 2nd harmonic (octave)
            0.4 * np.sin(2 * np.pi * freq * 3 * t),                # 3rd harmonic
            0.25 * np.sin(2 * np.pi * freq * 4 * t),               # 4th harmonic
            0.2 * np.sin(2 * np.pi * freq * 5 * t),                # 5th harmonic
            0.15 * np.sin(2 * np.pi * freq * 6 * t)                # 6th harmonic
        ]
        
        # Combine strings and harmonics
        wave = sum(strings) * 0.3 + sum(harmonics) * 0.7
        
        # Piano attack is fast with a quick decay and long release
        attack_time = 0.005  # Very fast attack
        decay_time = 0.1 + (1 - intensity) * 0.2  # Higher intensity = faster decay
        sustain_level = 0.3
        release_time = 0.5
        
        # Time points for ADSR envelope
        t_attack = min(attack_time, len(t)/44100) 
        t_decay = min(t_attack + decay_time, len(t)/44100)
        t_release = max(0, len(t)/44100 - release_time)
        
        # Create ADSR envelope
        envelope = np.zeros_like(t)
        attack_mask = t < t_attack
        decay_mask = (t >= t_attack) & (t < t_decay)
        sustain_mask = (t >= t_decay) & (t < t_release)
        release_mask = t >= t_release
        
        envelope[attack_mask] = t[attack_mask] / t_attack
        envelope[decay_mask] = 1 - (1 - sustain_level) * (t[decay_mask] - t_attack) / (t_decay - t_attack)
        envelope[sustain_mask] = sustain_level * np.exp(-(t[sustain_mask] - t_decay) / (0.5 + intensity * 1.0))
        envelope[release_mask] = sustain_level * np.exp(-(t[release_mask] - t_decay) / (0.5 + intensity * 1.0)) * (1 - (t[release_mask] - t_release) / (len(t)/44100 - t_release))
        
        # Apply envelope
        wave = wave * envelope
        
        # Piano resonance - body resonance frequencies
        resonance_freqs = [80, 120, 240, 450, 600]
        resonance = np.zeros_like(t)
        for res_freq in resonance_freqs:
            resonance += 0.01 * np.sin(2 * np.pi * res_freq * t) * np.exp(-t * 5)
        
        wave += resonance * intensity * 0.2
        
        # Piano hammer noise (attack)
        hammer_noise = np.random.randn(int(0.01 * 44100))  # 10ms of noise
        if len(hammer_noise) < len(wave):
            noise_pad = np.zeros(len(wave) - len(hammer_noise))
            hammer_noise = np.concatenate([hammer_noise, noise_pad])
        else:
            hammer_noise = hammer_noise[:len(wave)]
            
        # Add hammer noise at attack
        wave += hammer_noise * 0.05 * intensity * envelope
        
        # String resonance - sympathetic vibration
        sympathetic = 0.02 * np.sin(2 * np.pi * freq * 1.5 * t) * np.exp(-t * 2)
        wave += sympathetic * intensity
        
        return np.tanh(wave * (0.8 + intensity * 0.4)) 