import cv2

# Ouvrir la caméra
cap = cv2.VideoCapture(0)  # 0 indique la caméra par défaut. Si vous avez plusieurs caméras, vous pouvez essayer 1, 2, etc.

# Vérifier si la caméra est ouverte
if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la caméra.")
    exit()

while True:
    
    # Lire une frame depuis la caméra
    ret, frame = cap.read()

    # Changer la couleur de la frame en gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Retourner la video horizontalement
    gray = cv2.flip(gray, 1)

    # Vérifier si la frame a été lue correctement
    if not ret:
        print("Erreur: Impossible de lire la frame depuis la caméra.")
        break

    # Afficher la frame en gris
    cv2.imshow('Camera', gray)

    # Attendre 1 milliseconde et vérifier si l'utilisateur appuie sur la touche 'q' pour quitter
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break


# Libérer la caméra et fermer la fenêtre
cap.release()
cv2.destroyAllWindows()