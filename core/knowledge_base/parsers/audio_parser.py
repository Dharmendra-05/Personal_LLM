# ==== core/knowledge_base/parsers/audio_parser.py ====
"""
Audio Document Parser.

Extracts text transcripts from audio files using OpenAI's Whisper (local).
"""

from pathlib import Path
from core.utils.logger import get_logger

logger = get_logger(__name__)

def parse_audio(file_path: Path, model_size: str = "base") -> str | None:
    """Extract text from an audio file using local Whisper.

    Args:
        file_path: Path to the audio file (.mp3, .wav, .m4a).
        model_size: Whisper model size to use ('tiny', 'base', 'small', 'medium', 'large').

    Returns:
        Extracted audio transcript, or None if extraction fails.
    """
    try:
        import whisper
        import warnings
        
        # Suppress FP16 warnings on CPU
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logger.debug("Loading Whisper model '%s' for %s", model_size, file_path.name)
            model = whisper.load_model(model_size)
            
            logger.debug("Transcribing %s", file_path.name)
            result = model.transcribe(str(file_path))
            
            transcript = result.get("text", "").strip()
            
            if transcript:
                return f"--- Audio Transcript: {file_path.name} ---\n{transcript}"
                
            return None
            
    except ImportError:
        logger.error("Whisper is not installed. Run `pip install openai-whisper` and ensure `ffmpeg` is installed.")
        return None
    except Exception as exc:
        logger.error("Failed to parse Audio '%s': %s", file_path.name, exc)
        return None
