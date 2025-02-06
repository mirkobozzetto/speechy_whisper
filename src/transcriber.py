import numpy as np
import soundfile as sf
import librosa
import whisper
import traceback
from pydub import AudioSegment
import io

class AudioTranscriber:
	def __init__(self):
		# Load the Whisper model; depending on your server resources, you can
		self.whisper_model = whisper.load_model("large")

	def transcribe_audio(self, audio_data):
		print("Transcription en cours...")
		result = self.whisper_model.transcribe(audio_data)
		transcription = result["text"]
		print(f"Transcription: {transcription}")
		return transcription

	def process_audio(self, audio_stream):
		try:
			print("Démarrage du traitement audio...")
			# Convert the audio stream to a WAV file
			audio = AudioSegment.from_file(audio_stream, format="webm")
			wav_data = io.BytesIO()
			audio.export(wav_data, format="wav")
			wav_data.seek(0)

			# Read the WAV data
			audio_data, sample_rate = sf.read(wav_data)
			print(f"Audio chargé. Forme: {audio_data.shape}, Fréquence d'échantillonnage: {sample_rate}")

			if audio_data.ndim > 1:
				audio_data = audio_data.mean(axis=1)
				print("Converti en mono")

			if sample_rate != 16000:
				audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)
				print("Rééchantillonné à 16000 Hz")

			audio_data = audio_data.astype(np.float32)
			print("Converti en float32")

			transcription = self.transcribe_audio(audio_data)
			print(f"Transcription terminée: {transcription}")

			return transcription
		except Exception as e:
			print(f"Erreur dans process_audio: {str(e)}")
			print(traceback.format_exc())
			return None
