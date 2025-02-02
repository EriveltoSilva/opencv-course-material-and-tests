import os
import cv2
import numpy as np
from PIL import Image


def get_images_paths():
    images = [os.path.join("../../resources/yalefaces/train", filename)  for filename in os.listdir("../../resources/yalefaces/train")]
    #print("Imagens:\n")
    #print(images)
    return images


def get_ids_and_faces_images():
    images_paths = get_images_paths()
    faces = []
    ids = []

    for image_path in images_paths:
        # Carregar imagem e converter em matrix numpy
        faceImg = Image.open(image_path).convert('L') # cria e convert na escala de cinza
        face_np = np.array(faceImg, 'uint8') # converte a imagem em formato array do numpy

        # Processar id da imagem a partir do path
        id = int(image_path.split("/train/")[1].split(".")[0].replace("subject", ""))

        faces.append(face_np)
        ids.append(id)

    return ids, faces


def make_classifier(ids: list, faces: list):
    """ make yml face recognition classifier"""
    #lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
    lbph_classifier = cv2.face.LBPHFaceRecognizer.create()
    lbph_classifier.train(faces, ids)
    lbph_classifier.write("lbph_classifier.yml")


def get_user_Id(image_path) -> int:
    """ get subject id from image path"""
    return int(image_path.split("/train/")[1].split(".")[0].replace("subject", ""))


def main():
    ids, faces = get_ids_and_faces_images()

    make_classifier(ids, faces)

    """lbph_classifier = cv2.face.LBPHFaceRecognizer.create()
    lbph_classifier.read("lbph_classifier.yml")

    image_test_path = "../../resources/yalefaces/test/subject01.happy.gif"
    image_test = Image.open(image_test_path).convert('L')
    image_test_np = np.array(image_test, 'uint8')

    prediction = lbph_classifier.predict(image_test_np)
    print(prediction)

    cv2.putText(image_test_np, f"Pred:{prediction[0]}", (10,30), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0))
    cv2.putText(image_test_np, f"Exp:{get_user_Id(image_test_path)}", (10,30), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0))
    cv2.imshow("Image in Analises", image_test_np)
    """



if __name__ == '__main__':
    main()