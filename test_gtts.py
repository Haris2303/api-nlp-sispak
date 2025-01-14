from gtts import gTTS

sentence = "Halo bro!, Ko mau masuk ke fanlob ka trada?"
language = "id"

file = gTTS(text=sentence, lang=language)

file.save("halo.mp3")
