import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import csv

class Application:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Capture vidéo avec OpenCV
        self.cap = cv2.VideoCapture(0)

        # Création d'un canvas pour afficher la vidéo
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # Création d'un panneau à droite avec un dropdown, une entrée de texte et un bouton
        self.panel = tk.Frame(window)
        self.panel.grid(row=0, column=1, padx=10, pady=10)

        # Dropdown
        self.dropdown_label = tk.Label(self.panel, text="Liste des signes")
        self.dropdown_label.grid(row=0, column=0, pady=10)
        self.options = self.load_options_from_csv("keypoint_classifier_label.csv")
        self.dropdown_var = tk.StringVar()
        self.dropdown = ttk.Combobox(self.panel, values=self.options, textvariable=self.dropdown_var)
        self.dropdown.grid(row=1, column=0, pady=10)
        self.dropdown.set(self.options[0])  # Sélectionne la première option par défaut

        # Entrée de texte
        self.text_entry_label = tk.Label(self.panel, text="Entrée de texte:")
        self.text_entry_label.grid(row=2, column=0, pady=10)
        self.text_entry_var = tk.StringVar()
        self.text_entry = tk.Entry(self.panel, textvariable=self.text_entry_var)
        self.text_entry.grid(row=3, column=0, pady=10)

        # Bouton
        self.btn1 = tk.Button(self.panel, text="Bouton 1", command=self.on_button1_click)
        self.btn1.grid(row=4, column=0, pady=10)

        # Mise à jour de la vidéo
        self.update()

        # Fermeture propre de la fenêtre
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_options_from_csv(self, filename):
        options = []
        try:
            with open(filename, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    options.extend(row)
        except FileNotFoundError:
            print(f"Fichier CSV {filename} introuvable.")
        return options

    def update(self):
        # Capture d'une image depuis la caméra
        ret, frame = self.cap.read()
        if ret:
            # Conversion de l'image OpenCV en format compatible avec Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            # Affichage de l'image sur le canvas
            self.canvas.img = img
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)

            # Appel récursif pour mettre à jour l'affichage
            self.window.after(10, self.update)

    def on_button1_click(self):
        # Récupération du texte de l'entrée et de la sélection du dropdown
        text_entry_value = self.text_entry_var.get()
        dropdown_value = self.dropdown_var.get()
        print(f"Bouton 1 cliqué avec le texte : {text_entry_value}, sélection : {dropdown_value}")

    def on_closing(self):
        # Arrêt de la capture vidéo et fermeture de la fenêtre
        self.cap.release()
        self.window.destroy()

# Création de la fenêtre principale
root = tk.Tk()
app = Application(root, "SignLingo Interface DEV")
root.mainloop()
