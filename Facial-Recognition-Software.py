#Israel Showell
#2-7-2024
#File name: Facial-Recognition-Software.py
#Project:
#Facial Recognition Software
#Version - 1.0.0


import argparse

from pathlib import Path

from collections import Counter

import face_recognition

import pickle

from PIL import Image, ImageDraw

#Constant that holds the path where our pickled encodings will go
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

#Constant that holds the path where our input image is
#Used to test before command line
INPUT_IMAGE = Path("D:/Practice - Machine Learning/Python Projects/Face_Recognition_Tool/Face_Recognizer/unknown.jpg")

#Constants for the Bounding Box color and text color
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

#Desribes the command-line program
parser = argparse.ArgumentParser(description="Recognize faces in an image")

#Adds arguments to use the program
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument("--validate", action="store_true", help="Validate trained model")
parser.add_argument("--test", action="store_true", help="Test the model with an unknown image")
parser.add_argument("-m", action="store", default="hog", choices=["hog", "cnn"], help="Which model to use for training: hog (CPU), cnn (GPU)",)

#Asks for the input_image
parser.add_argument("-f", action="store", help="Path to an image with an unknown face")

#Adds the arguements
args = parser.parse_args()


def validate(model: str = "hog"):
    for filepath in Path("Validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(image_location=str(filepath.absolute()), model=model)
#End of function

"""Used to make a square to draw arounnd a face"""
def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )
#End of function

    
#Creates folders in the directory where this Script is!
#Will only do so if they don't exist already
Path("Training").mkdir(exist_ok=True)
Path("Output").mkdir(exist_ok=True)
Path("Validation").mkdir(exist_ok=True)

#Used to see where the path of the file you are printing out is!
print(INPUT_IMAGE)

def recognize_faces(
    image_location: str, #Location of the image we want to recognize
    model: str = "hog", #What model we are using
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
    #The location of where the encodes are
) -> None:

    #Opens the pickle file in read mode in bytes
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
    #This takes our input image and its location, and loads it into the Script
    input_image = face_recognition.load_image_file(image_location)
    """"This detects faces in the input image
and returns a list of tuples, where each tuple contains the
coordinates of a detected face in the format (top, right, bottom, left).
These coordinates represent the bounding box around each detected face."""
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    print(face_recognition.face_locations(input_image, model=model))
    """The face encodings are a numerical representation of the face's features,
The result is a list of face encodings, where each element
corresponds to one of the detected faces in the input image."""
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    
    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)
    
    """ A loop that iterates over pairs of elements from two lists simultaneously using the zip function.
input_face_locations: This list contains tuples representing the bounding boxes of detected faces in the input image.
input_face_encodings: This list contains the face encodings computed for each detected face in the input image.
bounding_box and unknown_encoding are loop variables that will hold the values from each iteration, representing the bounding box and face encoding of each detected face"""
    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "unknown"
# Removed print(name, bounding_box)
        _display_face(draw, bounding_box, name)
    del draw
    pillow_image.show()
#End of function
        
#Function for encoding faces
def encode_known_faces(
    #The model is hog
    #HOG (histogram of oriented gradients) is a common technique for object detection.
    #Works best with a CPU
    #Sends encodings to the Path stored in the constant
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None: #None means no value is returned!
    #Two empty lists
    names = []
    encodings = []

    #So, this loop will check the training directory where the images are
    #And will check each subfolder in the training directory    
    for filepath in Path("training").glob("*/*"):
        #For each file, it extracts the name of the parent directory (which presumably represents the name of the person)
        name = filepath.parent.name
        #Loads the image using face_recognition.load_image_file()
        image = face_recognition.load_image_file(filepath)

        #Detects the face in each image and get its encoding.
        #This is an array of numbers describing the features of the face,
        #and it’s used with the main model underlying face_recognition
        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        #Adds the name and encodings to the empty lists
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)
            
    #This is a dictionary that matches each name with its correct encoding
    name_encodings = {"names": names, "encodings": encodings}

    #Opens the Path stored in the constant
    #This opens the encodings.pkl file in write mode in bytes
    with encodings_location.open(mode="wb") as f:
        #Serializes the data
        pickle.dump(name_encodings, f)
#End of function





def _recognize_face(unknown_encoding, loaded_encodings):
    # Step 1: Compares the unknown face encoding with the known encodings
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)

    # Step 2: Count votes for each name based on matches
    votes = Counter(
        name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)

    # Step 3: Determine the most common recognized name
    if votes:
        return votes.most_common(1)[0][0]
#End of function


#Used for the args in the command line
if __name__ == "__main__":
    if args.train:
        #Runs the encoding function
        encode_known_faces(model=args.m)
    if args.validate:
        #Runs the validate function 
        validate(model=args.m)
    if args.test:
        #Runs the recognize function
        recognize_faces(image_location=args.f, model=args.m)

