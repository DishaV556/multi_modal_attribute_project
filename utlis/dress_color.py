import numpy as np

def detect_dress_color(img):
    h, w, _ = img.shape
    lower = img[int(h*0.6):h, :]
    avg = np.mean(lower, axis=(0,1))

    b, g, r = avg
    if r > g and r > b:
        return "Red"
    elif g > r and g > b:
        return "Green"
    elif b > r and b > g:
        return "Blue"
    else:
        return "Mixed"
