import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import pyvirtualcam

class FaceDetectionApp:
    def __init__(self, root, video_source=0):
        self.root = root
        self.root.title("Face Detection with Pixelation")

        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source)
        
        # Imposta la risoluzione al massimo supportato
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Aggiungi l'impostazione del frame rate (60 fps)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        self.virtual_cam = pyvirtualcam.Camera(width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                               height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                               fps=60)

        self.canvas = tk.Canvas(root, width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.canvas.pack()

        self.pixelate_button = ttk.Button(root, text="Pixelate", command=self.toggle_pixelation)
        self.pixelate_button.pack(pady=10)

        self.pixelate_faces = False

        # Utilizza il classificatore Haarcascades per il rilevamento dei volti
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.update()

    def toggle_pixelation(self):
        self.pixelate_faces = not self.pixelate_faces

    def update(self):
        ret, frame = self.cap.read()

        if ret:
            if self.pixelate_faces:
                frame = self.pixelate_faces_in_frame(frame)

            self.photo = self.convert_frame_to_photo(frame)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # Invia il frame alla webcam virtuale
            self.virtual_cam.send(frame)

        self.root.after(10, self.update)

    def convert_frame_to_photo(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=img)
        return photo

    def pixelate_faces_in_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            if self.pixelate_faces:
                face_roi = frame[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (w // 20, h // 20), interpolation=cv2.INTER_LINEAR)
                face_roi = cv2.resize(face_roi, (w, h), interpolation=cv2.INTER_NEAREST)
                frame[y:y+h, x:x+w] = face_roi
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        if self.virtual_cam.is_open():
            self.virtual_cam.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
