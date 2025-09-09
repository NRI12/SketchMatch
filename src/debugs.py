import cv2

image_path = r"C:\Users\nguye\Desktop\sketch-to-image\data\sketch_dataset\sketch\pineapple\n07753275_137-1.png"
img = cv2.imread(image_path)

if img is not None:
    print("Shape of the image:", img.shape)
else:
    print("Cannot read the image. Check the file path.")