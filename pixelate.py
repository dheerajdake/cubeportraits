import cv2
import math
from collections import Counter
import numpy as np
import os

# Properties
SCALE = 49    # Higher number, lower res
DIES_PER_CUBE = 3    # 3x3 cube
CUBE_DIE_MAGNIFICATION = 30    # Higher number, higher res cube face images

# File names
FILE_INPUT_IMAGE = "katy.jpg"
FILE_PIXELATED_IMAGE = FILE_INPUT_IMAGE[:-4] + "_pixelated.jpg"
FILE_CUBE_COLORED_IMAGE = FILE_INPUT_IMAGE[:-4] + "_cubefied.jpg"
FILE_CUBE_FACE_IMAGE_PREFIX = FILE_INPUT_IMAGE[:-4] + "_"

CUBE_FACES_FOLDER = "faces/"

# RGB values of RUBIX Cube as a list
# https://www.schemecolor.com/rubik-cube-colors.php#:~:text=The%20Rubik%20Cube%20Colors%20Color,created%20by%20user%20cosme%20damiao.
# Open CV saves as BRG, the below is in BRG format
rubixCubeColors = [
    [72, 155, 0],
    [0, 0, 185],
    [173, 69, 0],
    [0, 89, 255],
    [255, 255, 255],
    [0, 213, 255]
]

# Returns the pixel's RGB value
def getPixel(image, x, y):
    return image[x][y]


# Prints the pixel's RGB value
def printPixel(image, x, y):
    print(getPixel(image, x, y))


# Prints the Grid values
def printGrid(image, x1, y1, x2, y2):
    for i in range(x1, x2):
        for j in range(y1, y2):
            printPixel(image, i, j)


# Returns the most repeating element of the list
# If there are no repeating elements, returns the average
# If there is a tie, it returns the average of the tied elements
def getRepeatingPixel(plist):
    value = 0
    occurence_count = Counter(plist)
    #print("@@ {}".format(occurence_count))
    mode = occurence_count.most_common(1)[0][1]

    if (mode == 1):
        #print("Averaging...")
        value = sum(plist) / len(plist)
    else:
        # See if there are any elements with the same count
        selectedFreq = occurence_count.most_common(1)[0][1]

        matchCount = 0
        for k, v in occurence_count.items():
            if (v == selectedFreq):
                value += k
                matchCount = matchCount + 1

        value /= matchCount

    return int(value)


# Returns a list containing RGB values
def getGridMode(image, x1, y1, x2, y2, rows, cols):
    redElements = []
    greenElements = []
    blueElements = []

    for i in range(x1, x2):
        for j in range(y1, y2):
            if (x1 > rows or y1 > rows):
                pixelValue = getPixel(image, rows - 1, cols - 1)
            elif (x2 > rows or y2 > rows):
                pixelValue = getPixel(image, rows - 1, cols - 1)
            else:
                pixelValue = getPixel(image, i, j)

            redElements.append(pixelValue[0])
            greenElements.append(pixelValue[1])
            blueElements.append(pixelValue[2])        

    # Get mode
    redVal   = getRepeatingPixel(redElements)
    greenVal = getRepeatingPixel(greenElements)
    blueVal  = getRepeatingPixel(blueElements)

    rgbModeList = []
    rgbModeList.append(redVal)
    rgbModeList.append(greenVal)
    rgbModeList.append(blueVal)

    return rgbModeList


# Returns the closest rubix cube color match to inColor
closestColorDict = {}
def getRubixCubeColor(inColor):
    # Check pre-computed dict
    global closestColorDict

    # list can't be used as key, compute list string
    listString = "{}{}{}".format(inColor[0], inColor[1], inColor[2])

    if listString not in closestColorDict.keys():
        diffList = []

        for rbxclr in rubixCubeColors:
            dist = ((inColor[0] - rbxclr[0]) * (inColor[0] - rbxclr[0])) +\
                    ((inColor[1] - rbxclr[1]) * (inColor[1] - rbxclr[1])) +\
                    ((inColor[2] - rbxclr[2]) * (inColor[2] - rbxclr[2]))

            dist = math.sqrt(dist)
            diffList.append(dist)

            # Add to dict
            closestColorDict[listString] = rubixCubeColors[diffList.index(min(diffList))]

    return closestColorDict[listString]


# Replaces original color with closest Rubix Cube colors
def cubefyImage(image):
    print("Cubefying image...")
    rows = image.shape[0]
    cols = image.shape[1]

    for x in range(rows):
        for y in range(cols):
            image[x][y] = getRubixCubeColor(getPixel(image, x, y))

    return image


# Generate cube faces
def generateCubeFaces(image, root):
    rows = image.shape[0]
    cols = image.shape[1]

    # Cube properties
    PIXELS_PER_CUBE_DIE = root

    # Output file properties
    CUBE_FACE_RES = DIES_PER_CUBE * CUBE_DIE_MAGNIFICATION
    IMAGE_PIXELS_PER_CUBE = DIES_PER_CUBE * PIXELS_PER_CUBE_DIE

    cubeDiesInRow = rows / PIXELS_PER_CUBE_DIE
    cubesInRow = round(cubeDiesInRow / DIES_PER_CUBE)

    cubeDiesInCol = cols / PIXELS_PER_CUBE_DIE
    cubesInCol = round(cubeDiesInCol / DIES_PER_CUBE)

    totalCubes = cubesInRow * cubesInCol

    print("Row Cube Count: {} Col Cube Count: {}".format(cubesInRow, cubesInCol))
    print("Total {}x{} cubes needed = {}".format(DIES_PER_CUBE, DIES_PER_CUBE, totalCubes))
    print("Generating cube faces...")

    # Create pre-reqs
    if not os.path.exists(CUBE_FACES_FOLDER):
        os.mkdir(CUBE_FACES_FOLDER)

    # Generate face for each cube
    cubeFaceImage = np.zeros((CUBE_FACE_RES, CUBE_FACE_RES, 3), np.uint8)

    for i in range(0, cubesInRow):
        for j in range(0, cubesInCol):
            
            ix = i * IMAGE_PIXELS_PER_CUBE
            iy = j * IMAGE_PIXELS_PER_CUBE

            for a in range(0, CUBE_FACE_RES):
                for b in range(0, CUBE_FACE_RES):
                    _ix = ix + (int(a/CUBE_DIE_MAGNIFICATION) * root)
                    _iy = iy + (int(b/CUBE_DIE_MAGNIFICATION) * root)

                    # Handle out of bounds
                    if (_ix >= rows):
                        _ix = rows - 1

                    if (_iy >= cols):
                        _iy = cols - 1

                    cubeFaceImage[a][b] = getPixel(image, _ix, _iy)

            # Save image
            index = (cubesInCol * i) + j
            genFileName = CUBE_FACES_FOLDER + FILE_CUBE_FACE_IMAGE_PREFIX + str(index) + ".jpg"
            cv2.imwrite(genFileName, cubeFaceImage)
            print("Faces: {}/{}\r".format(index, totalCubes), end='')

    print("{} Cube faces generated".format(totalCubes))


# Scan image and pixelate
def pixelateImage(image, pixBlockCount):
    root = math.sqrt(pixBlockCount)

    # Round off to the nearest root
    if not root.is_integer():
        print("Not a perfect root")
        root = round(root)
        print("Re-adjusted to {}".format(root))
    else:
        root = int(root)

    # Compute pixel blocks based on image dimensions
    rows = image.shape[0]
    cols = image.shape[1]
    print("Image: {} x {}".format(rows, cols))

    rowBlocks = rows / root
    colBlocks = cols / root

    if not rowBlocks.is_integer():
        rowBlocks = int(rowBlocks) + 1
    else:
        rowBlocks = int(rowBlocks)
    
    if not colBlocks.is_integer():
        colBlocks = int(colBlocks) + 1
    else:
        colBlocks = int(colBlocks)

    print("RowBlocks: {} ColBlocks: {}".format(rowBlocks, colBlocks))
    print("Pixelating image...")

    # This is a list of lists containing the pixelated image
    imageList = []

    for i in range(0, rowBlocks):
        rowList = []
        imageList.append([])

        for j in range(0, colBlocks):
            x1 = i * root
            y1 = j * root
            x2 = x1 + root - 1
            y2 = y1 + root - 1

            # Add to row list
            rowList.append(getGridMode(image, x1, y1, x2, y2, rows, cols))

        # Merge with already appended empty list
        imageList[i].extend(rowList)

    # Output Image should be of the same resolution
    output_image = np.zeros((rows, cols, 3), np.uint8)

    # Generate pixelated image
    for i in range(0, rows):
        for j in range(0, cols):
            output_image[i][j] = imageList[int(i/root)][int(j/root)]

    # Save pixelated image
    cv2.imwrite(FILE_PIXELATED_IMAGE, output_image)
    print("Pixelated image generated")

    # Fill closest cube colors
    output_image = cubefyImage(output_image)

    # Save cubefied image
    cv2.imwrite(FILE_CUBE_COLORED_IMAGE, output_image)
    print("Cubed image generated")

    # Create cube faces based on cubefied image
    generateCubeFaces(output_image, root)



# Main
def main():
    image = cv2.imread(FILE_INPUT_IMAGE)
    pixelateImage(image, SCALE)

# Start here
main()
