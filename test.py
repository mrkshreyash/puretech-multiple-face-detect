import os

test_images = 'test_images/'

for files in os.listdir(test_images):
    print(os.path.join(test_images, files))