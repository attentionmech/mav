"""
Implements chip-tune style audio generation for token feedback.
"""
import numpy as np

class ChipTunePlayer:
    """Generates and plays chip-tune style sounds for token generation feedback."""
    
    def __init__(self):
        """Initialize the pygame mixer for audio playback."""
        self.initialized = False
        try:
            import pygame
            pygame.mixer.pre_init(frequency=44100, size=-16, channels=1)
            pygame.mixer.init()
            self.pygame = pygame
            self.initialized = True
        except ImportError:
            print("Warning: pygame not available. Audio feedback disabled.")
            self.initialized = False
    
    def play_chip_tone(self, token_id: int, entropy: float, duration: float = 0.1):
        """
        Play a chip-tune style sound based on token ID and entropy.
        
        Args:
            token_id: The token ID to map to pitch
            entropy: Attention entropy to map to volume (0-infinity)
            duration: Length of the sound in seconds
        """
        if not self.initialized:
            return
            
        # Map token_id to a musical pitch between C4–C6 (261–1047 Hz)
        base = 261.6  # C4
        freq = base * 2 ** ((token_id % 24) / 12)  # 2 octaves
        
        # Use entropy (0–something) to scale volume (clamp to [0–1])
        vol = float(min(max(entropy / 10, 0.0), 1.0))
        
        # Build a square wave (super retro) for 'duration' seconds
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        wave = 0.5 * np.sign(np.sin(2 * np.pi * freq * t))  # ±0.5
        
        # Convert to 16-bit PCM
        pcm = (wave * 32767).astype(np.int16)
        
        snd = self.pygame.sndarray.make_sound(pcm)
        snd.set_volume(vol)
        snd.play() 