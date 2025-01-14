from tkinter import*
import pyglet

root = Tk()

player = pyglet.media.Player()
song = "halo.mp3"
src = pyglet.media.load(song)
player.queue(src)
player.play()

root.mainloop()