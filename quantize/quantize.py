import cv2

image = cv2.imread('test.jpeg')

def quantize(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, quantized_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    black_color = 0
    quantized_image[quantized_image == black_color] = black_color
    rgba_image = cv2.cvtColor(quantized_image, cv2.COLOR_GRAY2BGRA)

    rgba_image[quantized_image == 255] = [0, 0, 0, 0]
    rgba_image[quantized_image == black_color] = [255,255,255,255]
    
    return rgba_image


res = quantize(image)

cv2.imwrite('result_image.png', res)