from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .transcriber import AudioTranscriber
import io

app = FastAPI(title="Speechy Whisper API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

transcriber = AudioTranscriber()

@app.post("/")
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        # Read the audio data from the uploaded file
        audio_stream = io.BytesIO(await audio.read())
        result = transcriber.process_audio(audio_stream)
        if result:
            return {
                "success": True,
                "transcription": result,
                "audio_file_name": audio.filename
            }
        else:
            raise HTTPException(status_code=500, detail="Échec de la transcription")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
