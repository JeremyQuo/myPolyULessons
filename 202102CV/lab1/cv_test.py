from PIL import Image # pillow package
import numpy as np

img = Image.open('a.jpg')               # read image file
arr = np.asarray(img, dtype=np.float64) # convert to np.ndarray

# process the array here, e.g.
arr = arr/2

arr = arr.astype(np.uint8) # make sure dtype is uint8
img = Image.fromarray(arr) # convert back to PIL.image object
img.save('b.jpg')          # save to file