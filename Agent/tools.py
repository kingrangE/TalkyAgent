import speech_recognition as sr
from playsound import playsound
from openai import OpenAI
from dotenv import load_dotenv
from gtts import gTTS as gt

load_dotenv()

r = sr.Recognizer()
mic = sr.Microphone()

# 음성 녹음 함수
def record_audio():
    with mic as source:
        print("recording...")
        audio_data = r.listen(source)
        print("complete!")
        return audio_data

def play_audio(file_name:str):
    playsound(file_name)

def stt(file_name:str):
    client = OpenAI()
    audio_file= open(file_name, "rb")

    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe", 
        file=audio_file
    )

    return transcription

def tts(content: str):
    speech = gt(content,lang="en")
    speech.save("ai_response.mp3")
    play_audio("ai_response.mp3")

if __name__ == "__main__":
    # # 음성 녹음
    # audio_data = record_audio()

    # # 오디오 파일로 저장
    # with open("test.wav", "wb") as f:
    #     f.write(audio_data.get_wav_data())

    # # 오디오 파일 재생
    # play_audio("test.wav")
    # print(stt("test.wav"))
    tts("hello. my name is kilwon. Have you ever studied how to use python?")