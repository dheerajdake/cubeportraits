import difflib
import cv2
import math
from collections import Counter
import numpy as np
import os
import sys


# LEGO Artboard configuration
LEGO_ART_BOARD_TILE_BLIPS_IN_X = 16
LEGO_ART_BOARD_TILE_BLIPS_IN_Y = 16

LEGO_ART_BOARD_TILES_IN_X = 3
LEGO_ART_BOARD_TILES_IN_Y = 3

LEGO_ART_BOARD_BLIPS_IN_X = LEGO_ART_BOARD_TILES_IN_X * LEGO_ART_BOARD_TILE_BLIPS_IN_X
LEGO_ART_BOARD_BLIPS_IN_Y = LEGO_ART_BOARD_TILES_IN_Y * LEGO_ART_BOARD_TILE_BLIPS_IN_Y

# LEGO Blips colors
LEGO_BLIPS = [
#      R    G    B  CNT  EXP
    [ 13,  16,  17,   0, 633], #1
    [ 44,  44,  44,   0, 431], #2
    [ 65,  65,  65,   0, 273], #3
    [130, 130, 130,   0, 292], #4
    [255, 255, 255,   0, 272], #5
    [217, 228, 220,   0, 199], #6
    [ 98, 216, 240,   0, 146], #7
    [  0,  69, 140,   0, 288], #8
    [  3,  18,  46,   0, 447], #9
    [ 77,  26,  10,   0, 172], #10
    [126,  45,   0,   0,  56], #11
    [218, 153, 114,   0, 158], #12
    [203, 171, 104,   0, 393], #13
    [  0, 123,  42,   0, 227], #14
    [120,  82, 133,   0,  97], #15
    [159,   0,  16,   0, 114]  #16
]

# Virtual LEGO artboard config
LEGOED_FACE_BLIP_RES_IN_X = 100  # In Pixels
LEGOED_FACE_BLIP_RES_IN_Y = 100

LEGOED_FACE_TILE_RES_IN_X = LEGOED_FACE_BLIP_RES_IN_X * LEGO_ART_BOARD_TILE_BLIPS_IN_X
LEGOED_FACE_TILE_RES_IN_Y = LEGOED_FACE_BLIP_RES_IN_Y * LEGO_ART_BOARD_TILE_BLIPS_IN_Y

LEGOED_FACE_IMAGE_RES_IN_X = LEGOED_FACE_TILE_RES_IN_X * LEGO_ART_BOARD_TILES_IN_X
LEGOED_FACE_IMAGE_RES_IN_Y = LEGOED_FACE_TILE_RES_IN_Y * LEGO_ART_BOARD_TILES_IN_Y

LEGOED_FACE_TILES_TO_OUTPUT = LEGO_ART_BOARD_TILES_IN_X * LEGO_ART_BOARD_TILES_IN_Y
LEGOED_FACE_IMAGE_TO_OUTPUT = 1



# Convert to artboard
def convertToArt(fileName, inputImage):
    inputImageRows = inputImage.shape[0]
    inputImageCols = inputImage.shape[1]

    print("Input Image Rows: {} Cols: {}".format(inputImageRows, inputImageCols))

    artRows = int(inputImageRows / (LEGO_ART_BOARD_BLIPS_IN_X))
    artCols = int(inputImageCols / (LEGO_ART_BOARD_BLIPS_IN_Y))

    print("art Image Rows: {} Cols: {}".format(artRows, artCols))

    # Pixelated res is artRows x artCols
    # Make them same, set lowest value to both to prevent overflow

    if artRows > artCols:
        artRows = artCols
    else:
        artCols = artRows

    print("art Image adjusted Rows: {} Cols: {}".format(artRows, artCols))
    artImage = []

    # Scan artRows x artCols and set every pixel in that area
    # to averaged pixel
    for rowPixel in range(0, LEGO_ART_BOARD_BLIPS_IN_X):
        artImage.append([])
        rowList = []

        for colPixel in range(0, LEGO_ART_BOARD_BLIPS_IN_Y):
            repeatedPixel_X = []
            repeatedPixel_Y = []
            repeatedPixel_Z = []

            # Get averaged pixel
            for avgRowPixel in range(0, artRows):
                for avgColPixel in range(0, artCols):
                    x_offset = (rowPixel * artRows) + avgRowPixel
                    y_offset = (colPixel * artCols) + avgColPixel
                    repeatedPixel_X.append(inputImage[x_offset][y_offset][0])
                    repeatedPixel_Y.append(inputImage[x_offset][y_offset][1])
                    repeatedPixel_Z.append(inputImage[x_offset][y_offset][2])

            # Get repeating pixel
            rPixel = max(set(repeatedPixel_X), key = repeatedPixel_X.count)
            gPixel = max(set(repeatedPixel_Y), key = repeatedPixel_Y.count)
            bPixel = max(set(repeatedPixel_Z), key = repeatedPixel_Z.count)

            rgbList = []
            rgbList.append(rPixel)
            rgbList.append(gPixel)
            rgbList.append(bPixel)

            # All add RGB values in cols to this rowList
            rowList.append(rgbList)

        artImage[rowPixel].extend(rowList)

    # Output images
    output_image = np.zeros((LEGO_ART_BOARD_BLIPS_IN_X * LEGOED_FACE_BLIP_RES_IN_X,
                    LEGO_ART_BOARD_BLIPS_IN_Y * LEGOED_FACE_BLIP_RES_IN_Y, 3), np.uint8)

    # Save realistic image
    print("Saving pixelated image")

    for rp in range(0, LEGO_ART_BOARD_BLIPS_IN_X * LEGOED_FACE_BLIP_RES_IN_X):
        for cp in range(0, LEGO_ART_BOARD_BLIPS_IN_Y * LEGOED_FACE_BLIP_RES_IN_Y):
            x_offset = int(rp/LEGOED_FACE_BLIP_RES_IN_X)
            y_offset = int(cp/LEGOED_FACE_BLIP_RES_IN_Y)
            output_image[rp][cp] = artImage[x_offset][y_offset]

    cv2.imwrite(fileName + "_real_pixelated.JPG", output_image)

    # At this point artImage has all the required info to build lego art
    # Replace realistic colors with lego blip colors
    for rowPixel in range(0, LEGO_ART_BOARD_BLIPS_IN_X):
        for colPixel in range(0, LEGO_ART_BOARD_BLIPS_IN_Y):
            diffList = []

            # Find distance for each color
            for legoIdx in range(0, len(LEGO_BLIPS)):
                rDst = (artImage[rowPixel][colPixel][0] - LEGO_BLIPS[legoIdx][0]) ** 2 
                gDst = (artImage[rowPixel][colPixel][1] - LEGO_BLIPS[legoIdx][1]) ** 2 
                bDst = (artImage[rowPixel][colPixel][2] - LEGO_BLIPS[legoIdx][2]) ** 2 
                diffList.append(math.sqrt(rDst + gDst + bDst))

            closeLegoColorIdx = diffList.index(min(diffList))

            # Increment blip counter - tells how many blips are required
            LEGO_BLIPS[closeLegoColorIdx][3] += 1

            # Save RGB values
            rgbList = []
            rgbList.append(LEGO_BLIPS[closeLegoColorIdx][0])
            rgbList.append(LEGO_BLIPS[closeLegoColorIdx][1])
            rgbList.append(LEGO_BLIPS[closeLegoColorIdx][2])

            # Replace color
            artImage[rowPixel][colPixel] = rgbList

    # Save legoed art
    print("Saving legoed image")

    for rp in range(0, LEGO_ART_BOARD_BLIPS_IN_X * LEGOED_FACE_BLIP_RES_IN_X):
        for cp in range(0, LEGO_ART_BOARD_BLIPS_IN_Y * LEGOED_FACE_BLIP_RES_IN_Y):
            x_offset = int(rp/LEGOED_FACE_BLIP_RES_IN_X)
            y_offset = int(cp/LEGOED_FACE_BLIP_RES_IN_Y)
            output_image[rp][cp] = artImage[x_offset][y_offset]

    cv2.imwrite(fileName + "_legoed_pixelated.JPG", output_image)

    # Generate lego tiles
    lego_tile_image = np.zeros((LEGO_ART_BOARD_TILE_BLIPS_IN_X * LEGOED_FACE_BLIP_RES_IN_X,
                    LEGO_ART_BOARD_TILE_BLIPS_IN_Y * LEGOED_FACE_BLIP_RES_IN_Y, 3), np.uint8)

    for tile in range(0, LEGO_ART_BOARD_TILES_IN_X * LEGO_ART_BOARD_TILES_IN_Y):
        print("Saving tile {}".format(tile))

        for rp in range(0, LEGO_ART_BOARD_TILE_BLIPS_IN_X * LEGOED_FACE_BLIP_RES_IN_X):
            for cp in range(0, LEGO_ART_BOARD_TILE_BLIPS_IN_Y * LEGOED_FACE_BLIP_RES_IN_Y):
                rp_offset = int(tile / 3) * (LEGO_ART_BOARD_TILE_BLIPS_IN_Y * LEGOED_FACE_BLIP_RES_IN_Y)
                cp_offset = int(tile % 3) * (LEGO_ART_BOARD_TILE_BLIPS_IN_X * LEGOED_FACE_BLIP_RES_IN_X)
                lego_tile_image[rp][cp] = output_image[rp_offset + rp][cp_offset + cp]
        
        cv2.imwrite(fileName + "_legoed_tile_{}.JPG".format(tile), lego_tile_image)



def main():
    fileName = sys.argv[1]
    print("Reading image: " + fileName)
    image = cv2.imread(fileName)
    convertToArt(fileName, image)

    totalPieces = 0

    for i in range(0, len(LEGO_BLIPS)):
        totalPieces += LEGO_BLIPS[i][3]
        print("Color {} Use {}".format(i+1, LEGO_BLIPS[i][3]))

    print("Total blips: {}".format(totalPieces))

### Program Entry ###
main()
