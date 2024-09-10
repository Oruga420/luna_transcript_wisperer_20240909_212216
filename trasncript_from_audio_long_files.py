import os
import sys
import gradio as gr
import openai
from pathlib import Path
from dotenv import load_dotenv
from pydub import AudioSegment
import tempfile

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

def compress_audio(audio_segment):
    return audio_segment.compress_dynamic_range()

def transcribe_chunk(chunk, filename):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        compressed_chunk = compress_audio(chunk)
        compressed_chunk.export(temp_file.name, format="mp3", bitrate="64k")
        
        # Check file size
        file_size = os.path.getsize(temp_file.name)
        if file_size > 25 * 1024 * 1024:  # If larger than 25MB
            raise ValueError(f"Compressed chunk size ({file_size / 1024 / 1024:.2f}MB) exceeds 25MB limit")
        
        with open(temp_file.name, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
    os.unlink(temp_file.name)
    return transcript["text"]

def transcribe_audio(audio_file):
    try:
        audio = AudioSegment.from_file(audio_file)
        chunk_length_ms = 5 * 60 * 1000  # 5 minutes in milliseconds
        chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        
        full_transcript = ""
        for i, chunk in enumerate(chunks):
            print(f"Transcribing chunk {i+1} of {len(chunks)}...")
            try:
                chunk_transcript = transcribe_chunk(chunk, f"chunk_{i}.mp3")
                full_transcript += chunk_transcript + " "
            except ValueError as e:
                print(f"Error processing chunk {i+1}: {str(e)}")
                continue
        
        # Generate a filename based on the original audio filename
        audio_filename = Path(audio_file).stem
        transcript_filename = f"{audio_filename}_transcript.txt"
        full_save_path = os.path.join(TRANSCRIPT_SAVE_PATH, transcript_filename)
        
        # Save the full transcript
        with open(full_save_path, "w", encoding="utf-8") as f:
            f.write(full_transcript.strip())
        
        return f"Transcription saved to: {full_save_path}\n\nTranscript:\n{full_transcript.strip()}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create the Gradio interface
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs="text",
    title="Audio Transcription with OpenAI Whisper",
    description="Upload an audio file to transcribe using OpenAI's Whisper API. Large files will be processed in chunks. The transcript will be saved in the specified folder."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()