from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import supervision as sv
import easyocr
import keras
import pickle

reader = easyocr.Reader(['en'], recognizer=False)
# YOLOv8_model = YOLO("../models/data-aug-obb-100.pt")
# YOLOv8_model = YOLO("../models/yolo-obb-light-100ep.pt")
YOLOv8_model = YOLO("../models/self-learning-130ep.pt")
# OCR_model = keras.models.load_model('../models/ocr_model.h5')
OCR_model = keras.models.load_model('../models/ocr_model_combined.h5')
# lb = pickle.load(open('../models/label_encoder.pkl', 'rb'))
lb = pickle.load(open('../models/label_encoder_combined.pkl', 'rb'))

def crop_license_plate(img):
    """
    Runs the YOLO model on the input image and crops the license plate

    Returns:
    - Image: the cropped license plate
    """
    results = YOLOv8_model(img, conf = 0.025, max_det = 1)

    detections = sv.Detections.from_ultralytics(results[0])
    xyxyxyxy = detections[0].data['xyxyxyxy'][0]

    point1, point2, point3, point4 = xyxyxyxy
    box_points = [point3, point2, point1, point4]

    if box_points[0][0] > box_points[1][0]:
        box_points = [point1, point4, point3, point2]

    # make blank image of size 520x110
    blank_image = np.zeros((110, 520, 3), np.uint8)
    blank_image[:] = (255, 255, 255)

    # warp
    matrix = cv2.getPerspectiveTransform(np.array(box_points, dtype=np.float32), np.array([[0, 0], [520, 0], [520, 110], [0, 110]], dtype=np.float32))
    warped = cv2.warpPerspective(cv2.imread(img), matrix, (520, 110))

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    warped = Image.fromarray(warped)
    return warped

def detect_text(img):
    """
    Detects text boxes in the input image

    Returns:
    - list: list of bounding boxes of the detected text
    """
    img = np.array(img)
    result = reader.detect(img, add_margin=0, slope_ths=0)[1][0]
    return result

def find_largest_text_area(results):
    """
    Finds the largest text area in the input image

    Returns:
    - list: list of bounding boxes of the largest text area
    """
    max_width = 0
    max_width_box = None
    for box in results:
        width = box[1][0] - box[0][0]
        if width > max_width:
            max_width = width
            max_width_box = box

    return max_width_box

def find_largest_text_area(results):
    """
    Finds the largest text area in the input image

    Returns:
    - list: list of bounding boxes of the largest text area
    """
    max_width = 0
    max_width_box = None
    for box in results:
        width = box[1][0] - box[0][0]
        if width > max_width:
            max_width = width
            max_width_box = box

    return max_width_box

def extract_box(img, result):
    """
    Extracts the bounding box of the detected text

    Returns:
    - Image: the cropped text box
    """
    p1, p2, p3, p4 = result
    box_points = [p1, p2, p3, p4]

    # make blank image of size 520x110
    size = (510, 110)
    blank_image = np.zeros((size[0], size[1], 3), np.uint8)
    blank_image[:] = (255, 255, 255)

    # warp
    matrix = cv2.getPerspectiveTransform(np.array(box_points, dtype=np.float32), np.array([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]], dtype=np.float32))
    warped = cv2.warpPerspective(img, matrix, (size[0], size[1]))

    warped = Image.fromarray(warped)

    return warped

def thresholding(img):
    """
    Applies thresholding to the input image
    """
    img.save("temp.jpg")
    img = cv2.imread("temp.jpg", cv2.IMREAD_GRAYSCALE)

    _, im_gray_th = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    im_gray_th = cv2.bitwise_not(im_gray_th)

    return Image.fromarray(im_gray_th)

def get_binary_lp_cutout(path_to_img):
    """
    Returns a binary image of the license plate cutout
    """
    cropped = crop_license_plate(path_to_img)
    result = detect_text(cropped)
    result = find_largest_text_area(result)

    return thresholding(extract_box(np.array(cropped), result))

def do_character_segmentation(img):
    """
    Does character segmentation on the input image

    Returns:
    - list: list of cropped character images
    - idx: index of the two smallest bounding boxes i.e the two stripes
    """
    img = np.array(img)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    boundingBoxes = sorted(boundingBoxes, key=lambda x: x[2]*x[3], reverse=True)[:8]
    boundingBoxes = sorted(boundingBoxes, key=lambda x: x[0])

    # Get two smallest bounding boxes
    boundingBoxesSmallest = sorted(boundingBoxes, key=lambda x: x[2]*x[3])[:2]
    
    # Get index of the two smallest bounding boxes
    idx1 = boundingBoxes.index(boundingBoxesSmallest[0])
    idx2 = boundingBoxes.index(boundingBoxesSmallest[1])


    characters = []
    for box in boundingBoxes:
        x, y, w, h = box
        characters.append(img[y:y+h, x:x+w])

    return characters, (idx1, idx2)

def predict_characters(characters, idx):
    """
    Predicts the characters in the input image

    Returns:
    - string: string of predicted characters
    """
    predicted_characters = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    for i, char in enumerate(characters):
        if i == idx[0] or i == idx[1]:
            predicted_characters[i] = '-'
            continue

        char = Image.fromarray(char)
        char = char.resize((20, 30))

        new_img = Image.new("RGB", (23, 33), "black")

        width, height = char.size
        x = 23//2 - width//2
        y = 33//2 - height//2

        new_img.paste(char, (x, y))
        new_img = new_img.convert('L')

        img_array = np.array(new_img)
        img_array = img_array / 255.0
        img_array = img_array.astype(int)
        img_array = img_array.reshape(1, 33, 23, 1)

        prediction = OCR_model.predict(img_array, verbose=0)
        prediction = np.array(prediction)
        preds = lb.inverse_transform(prediction.reshape(1, -1))
        predicted_characters[i] = preds[0]

    predicted_characters = ''.join(predicted_characters)

    return predicted_characters

def alpr(img):
    """
    Runs the ALPR pipeline on the input image

    Returns:
    - string: predicted license plate
    """
    binary_lp_cutout = get_binary_lp_cutout(img)
    characters, idx = do_character_segmentation(np.array(binary_lp_cutout))
    predicted_characters = predict_characters(characters, idx)

    return predicted_characters

