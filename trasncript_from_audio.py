import os
import sys
import gradio as gr
import openai
from pathlib import Path
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# Set up your OpenAI API key
openai.api_key = os.getenv('LUNAS_OPENAI_API_KEY')

if not openai.api_key:
    raise ValueError("LUNAS_OPENAI_API_KEY not found in environment variables.")

# Define the path where transcripts will be saved
TRANSCRIPT_SAVE_PATH = r"G:\Shared drives\Sesh\☀️ SeshWithFriends\PMO\Meetings"

# Set ffmpeg and ffprobe paths
FFMPEG_PATH = r"C:\Users\chuck\OneDrive\Desktop\Dev\ffmpeg\bin"
FFMPEG_EXECUTABLE = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
FFPROBE_EXECUTABLE = os.path.join(FFMPEG_PATH, "ffprobe.exe")

# Check if ffmpeg and ffprobe exist
if not os.path.exists(FFMPEG_EXECUTABLE):
    raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_EXECUTABLE}")
if not os.path.exists(FFPROBE_EXECUTABLE):
    raise FileNotFoundError(f"ffprobe not found at {FFPROBE_EXECUTABLE}")

# Add ffmpeg path to system PATH
os.environ["PATH"] += os.pathsep + FFMPEG_PATH

# Set the paths for pydub
AudioSegment.converter = FFMPEG_EXECUTABLE
AudioSegment.ffprobe = FFPROBE_EXECUTABLE

def transcribe_audio(audio_file):
    try:
        # Convert audio to WAV format
        audio = AudioSegment.from_file(audio_file)
        wav_path = audio_file + ".wav"
        audio.export(wav_path, format="wav")

        # Open the WAV file
        with open(wav_path, "rb") as file:
            # Call the OpenAI Whisper API
            transcript = openai.Audio.transcribe("whisper-1", file)
        
        # Clean up temporary WAV file
        os.remove(wav_path)
        
        # Extract the transcribed text
        transcribed_text = transcript["text"]
        
        # Generate a filename based on the original audio filename
        audio_filename = Path(audio_file).stem
        transcript_filename = f"{audio_filename}_transcript.txt"
        full_save_path = os.path.join(TRANSCRIPT_SAVE_PATH, transcript_filename)
        
        # Save the transcript
        with open(full_save_path, "w", encoding="utf-8") as f:
            f.write(transcribed_text)
        
        return f"Transcription saved to: {full_save_path}\n\nTranscript:\n{transcribed_text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create the Gradio interface
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs="text",
    title="Audio Transcription with OpenAI Whisper",
    description="Upload an audio file to transcribe using OpenAI's Whisper API. The transcript will be saved in the specified folder."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(share=True)