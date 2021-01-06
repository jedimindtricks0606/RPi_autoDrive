
import cv2
cv2.namedWindow("Resize Preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    print('Original Dimensions : ',frame.shape)
else:
    rval = False

width = 640
height = 480
dim = (width, height)
# resize image
resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ',resized.shape)

while rval:
    cv2.imshow("Resize Preview", cv2.flip(frame, 1))
    rval, frame = vc.read()
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("Resize Preview")

