import pyttsx3

# Membuat objek Engine
engine = pyttsx3.init()

# Mendapatkan daftar suara yang tersedia
voices = engine.getProperty('voices')

engine.setProperty("voice", voices[1].id)

# Mencari dan memilih suara Bahasa Indonesia
for voice in voices:
    print(f"ID: {voice.id}\nName: {voice.name}\nLanguages: {voice.languages}\n")

# Teks yang akan diucapkan
text = "Halo, selamat datang di konversi teks ke suara menggunakan Python dan pyttsx3."

# Mengatur kecepatan bicara dan volume (opsional)
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Mengucapkan teks
engine.say(text)

# Menjalankan perintah untuk mengucapkan
engine.runAndWait()
