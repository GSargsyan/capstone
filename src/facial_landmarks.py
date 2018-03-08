from module_imports import *

def progress(total, current):
    return round((total / (total - current)) * 100)

IMAGE_NAME = 'images/me.jpg'
error_levels = {1: 0.1, 10: 0.01, 100: 0.001, 1000: 0.0001}

golden_ratio = 1.618033988749895 
pp = pprint.PrettyPrinter(indent=4)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    """
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    """

    # show the face number
    """
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
	    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    """

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    """
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    """
    
    print("Setting distances...")
    distances = {}
    #distances = []
    length = len(shape)
    for i in range(length):
        pfrom = shape[i]
        distances[i] = {}
        for j in range(i + 1, length):
            pto = shape[j]
            dis = ((pto[0] - pfrom[0]) ** 2 + (pto[1] - pfrom[1]) ** 2) ** 0.5
            #distances.append((i, j, dis))
            distances[i][j] = dis

    pp.pprint(distances)
    ratios = []

    
    for p1_idx, vals1 in distances.items():
        for p2_idx, dis in vals1.items():
            for p3_idx, vals2 in distances.items():
                ratio = 0
                if p3_idx != p2_idx and p3_idx != p1_idx:
                    if p3_idx < p2_idx:
                        ratio = dis / vals2[p2_idx]
                    elif p3_idx == p2_idx:
                        ratios.extend([
                            (p1_idx, p2_idx, i, dis / v) for i, v in vals2.items()])
                if ratio != 0:
                    ratios.append((p1_idx, p2_idx, p3_idx, ratio))
    #pp.pprint(ratios) 
    total = 0
    for (p1_idx, p2_idx, p3_idx, r) in ratios:
        for score, error in error_levels.items():
            if abs(r - golden_ratio) < score:
                total += score
                rgb = tuple(random.randint(0, 255) for _ in range(3))

                p1 = tuple(shape[p1_idx])
                p2 = tuple(shape[p2_idx])
                p3 = tuple(shape[p3_idx])
                
                """
                cv2.line(image, p1, p2, rgb)
                cv2.line(image, p2, p3, rgb)
                """
    print("TOTAL: " + str(total))



    #for d, points in distances.items():
        

        #
 
# show the output image with the face detections + facial landmarks
lands = cv2.imread('landmarks_order.jpg')

lands = cv2.resize(lands, (600, 500))
#cv2.imshow("Landmarks", lands)

"""
cv2.imshow("Output", image)
cv2.waitKey(0)
"""
