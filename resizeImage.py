import cv2


def resizeImageScale(image, scale):
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    return resized


def resizeImageFix(image, width, height):
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized
