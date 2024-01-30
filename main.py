import face_recognition
import os
import shutil
import time
from colorama import Fore
import imghdr
from PIL import Image
import numpy as np
import re

# GLOBAL PARAMETERS
TOLERANCE = 0.5678
PERCENTAGE_TO_RESIZE = 1

# Program Initial Time
program_started = time.time()

print("System Started...\n")

print("TOLERANCE: {}, PERCENTAGE TO RESIZE: {}".format(TOLERANCE, PERCENTAGE_TO_RESIZE))


# Function to check if a person's face matches the reference image
def is_match(ref_encodings, face_encoding):
    return any(face_recognition.compare_faces([ref_encode], face_encoding, tolerance=TOLERANCE)[0] for ref_encode in
               ref_encodings)


def sanitize_filename(file_name):
    return re.sub(r'\x00', '', file_name)


print("\nWait for some time.... Creating Encodings for Target Image....\n")

# Load the reference image (the person you want to match)

reference_encodings = []

images = 'test_images_3/'
for files in os.listdir(images):
    image_path = os.path.join(images, files)
    reference_image = face_recognition.load_image_file(image_path)
    percentage = 1
    resized_height = int(reference_image.shape[0] * percentage)
    resized_width = int(reference_image.shape[1] * percentage)
    resized_image = np.array(Image.fromarray(reference_image).resize((resized_width, resized_height), Image.LANCZOS))
    ref_encoding = face_recognition.face_encodings(resized_image)[0]
    reference_encodings.append(ref_encoding)
    print(f"Encoding created and appended for filename :{image_path}")

# Directory containing images
images_directory = "female_face_matching/resized"

# Output directory for matched images
output_directory = "female_face_matching_0.5678"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

total_files, face_found, face_not_found, zero_faces_found, files_skipped = 0, 0, 0, 0, 0

for filename in os.listdir(images_directory):
    filename = sanitize_filename(filename)
    image_path = os.path.join(images_directory, filename)
    total_files += 1

    # Use 'imghdr' to check if the file is an image
    image_type = imghdr.what(image_path)

    if image_type not in ['jpeg', 'png', 'gif', 'bmp', 'jpg']:
        print(f"{Fore.YELLOW}Skipped file : {filename} (Possibly Not an image)")
        print(Fore.RESET)
        files_skipped += 1
        continue

    try:
        # initial time
        t0 = time.time()

        # Load the image
        current_image = face_recognition.load_image_file(image_path)

        # Print the shape of the original image
        print(f"\n==========> ACCESSED {filename} <==========")
        print(f"Original Image Shape: {current_image.shape}\n")

        # Resize the image
        percentage = PERCENTAGE_TO_RESIZE
        resized_height = int(current_image.shape[0] * percentage)
        resized_width = int(current_image.shape[1] * percentage)
        resized_image = np.array(Image.fromarray(current_image).resize((resized_width, resized_height), Image.LANCZOS))

        # Print the shape of the resized image
        print(f"Resized Image Shape: {resized_image.shape}\n")

        # Face detection on resized image
        face_locations = face_recognition.face_locations(resized_image)
        face_encodings = face_recognition.face_encodings(resized_image, face_locations)

        # Print the number of face encodings detected
        print(f"Processing {filename}: {len(face_encodings)} face(s) detected\n")

        for i, (top, right, bottom, left) in enumerate(face_locations):
            confidence = face_recognition.face_distance(reference_encodings, face_encodings[i])[0]
            print(f"Face {i + 1}: Confidence = {confidence:.4f}")

        # Check if a face is found in the image
        if len(face_encodings) > 0:
            # Check if the face matches the reference image
            if is_match(reference_encodings, face_encodings[0]):
                print(f"{Fore.GREEN}Face match found in {filename}")
                print(Fore.RESET, end="")
                face_found += 1

                # final time
                t1 = time.time()

                print(f"{Fore.BLUE}Time Taken to detect and process : {t1 - t0} sec.\n")
                print(Fore.RESET)

                # Copy the matched image to the output directory
                output_path = os.path.join(output_directory, filename)
                resized_image_dir = os.path.join(output_directory, "resized")
                resized_output_path = os.path.join(output_directory, "resized", filename)
                if not os.path.exists(resized_image_dir):
                    os.makedirs(resized_image_dir)
                Image.fromarray(resized_image).save(resized_output_path)

            else:
                print(f"{Fore.RED}Face match not found in {filename}\n")
                print(Fore.RESET, end="")
                face_not_found += 1

        else:
            print(f"{Fore.YELLOW}Zero Faces found in {filename}. Storing in null_directory.\n")
            print(Fore.RESET)
            zero_faces_found += 1
            null_directory = "null_directory"
            # if not os.path.exists(null_directory):
            #     os.makedirs(null_directory)
            # shutil.copy(image_path, os.path.join(null_directory, filename))

    except Exception as e:
        print(f"Error processing {filename}: {e}\n")

# ...

print("Matching process completed.")

print(f"""
TOLERANCE      : {TOLERANCE}
PER TO RESIZE  : {PERCENTAGE_TO_RESIZE}
TOTAL FILES    : {total_files}
FACE FOUND     : {face_found}
FACE NOT FOUND : {face_not_found}
SKIPPED FILES  : {files_skipped}
ZERO FACES     : {zero_faces_found}
""")

# Program Ended time
program_end = time.time()

print(f"\nTotal Time Taken : {(program_end - program_started)/60} Min(s)\n")
