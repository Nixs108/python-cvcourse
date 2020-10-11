# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')


# %%
img = cv2.imread('../DATA/sudoku.jpg', 0)


# %%
display_img(img)


# %%
sobolx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)


# %%
display_img(sobolx)


# %%
soboly = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)


# %%
display_img(soboly)


# %%
laplacian = cv2.Laplacian(img, cv2.CV_64F)


# %%
display_img(laplacian)


# %%
blended = cv2.addWeighted(src1=sobolx, alpha=0.5, src2=soboly, beta=0.5, gamma=0)


# %%
display_img(blended)


# %%
ret, th1 = cv2.threshold(src=img, thresh=127, maxval=255, type=cv2.THRESH_BINARY)


# %%
display_img(th1)


# %%
ret, th1 = cv2.threshold(src=blended, thresh=200, maxval=255, type=cv2.THRESH_BINARY_INV)


# %%
display_img(th1)


# %%
kernel = np.ones((4,4), np.uint8)
gradient = cv2.morphologyEx(src=blended, op=cv2.MORPH_GRADIENT, kernel=kernel)


# %%
display_img(gradient)


# %%
#histograms


# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
dark_horse = cv2.imread('../DATA/horse.jpg')
show_horse = cv2.cvtColor(dark_horse, cv2.COLOR_BGR2RGB)

rainbow = cv2.imread('../DATA/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2. COLOR_BGR2RGB)

blue_bricks = cv2.imread('../DATA/bricks.jpg')
show_bricks = cv2.cvtColor(blue_bricks, cv2.COLOR_BGR2RGB)


# %%
plt.imshow(dark_horse)


# %%
hist_val = cv2.calcHist(images=[blue_bricks], channels=[0], mask=None, histSize=[256], ranges=[0, 256])


# %%
hist_val.shape


# %%
plt.plot(hist_val)


# %%
color = ('b', 'g', 'r')


# %%
type(color)


# %%
for i, col in enumerate(color):
    hist = cv2.calcHist(images=[blue_bricks], channels=[i],mask=None,histSize=[256], ranges=[0, 256])
    plt.plot(hist, color=col)
plt.legend(color)
plt.show()
    


# %%
def show_histo_bgr(img):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist(images=[img], channels=[i],mask=None,histSize=[256], ranges=[0, 256])
        plt.plot(hist, color=col)
    plt.legend(color)
    plt.show()


# %%
show_histo_bgr(rainbow)


# %%
def show_histo_bgr_mask(img, mask=None):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist(images=[img], channels=[i],mask=mask,histSize=[256], ranges=[0, 256])
        plt.plot(hist, color=col)
    plt.legend(color)
    plt.show()


# %%
show_histo_bgr(rainbow)


# %%
img = rainbow


# %%
img.shape


# %%
img.shape[:2]


# %%
mask = np.zeros(img.shape[:2], dtype=np.uint8)


# %%
#plt.imshow(mask, cmap='gray')
mask[300:400, 100:400] = 255


# %%
plt.imshow(mask, cmap='gray')


# %%
plt.imshow(show_rainbow)


# %%
mask_img = cv2.bitwise_and(src1=img, src2=img, mask=mask)
mask_img_show = cv2.bitwise_and(src1=show_rainbow, src2=show_rainbow, mask=mask)


# %%
plt.imshow(mask_img)


# %%
plt.imshow(mask_img_show)


# %%
hist = cv2.calcHist(images=[rainbow], channels=[2], mask=None, histSize=[256], ranges=[0, 256])
plt.plot(hist)


# %%
hist = cv2.calcHist(images=[rainbow], channels=[2], mask=mask, histSize=[256], ranges=[0, 256])
plt.plot(hist)


# %%
show_histo_bgr(rainbow)


# %%
show_histo_bgr_mask(rainbow, mask=mask)


# %%
gorilla = cv2.imread('../DATA/gorilla.jpg', 0)


# %%
plt.imshow(gorilla, cmap='gray')


# %%
def display_img(img, cmap=None):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap)


# %%
display_img(gorilla, cmap='gray')


# %%
gorilla.shape


# %%
hist_values = cv2.calcHist(images=[gorilla], channels=[0], mask=None, histSize=[256], ranges=[0,256])


# %%
plt.plot(hist_values)


# %%
eq_gorilla = cv2.equalizeHist(gorilla)
display_img(eq_gorilla, cmap='gray')


# %%
hist_values_eq = cv2.calcHist(images=[eq_gorilla], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.plot(hist_values_eq)


# %%
color_gorilla = cv2.imread('../DATA/gorilla.jpg')
show_color_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2RGB)


# %%
display_img(show_color_gorilla)


# %%
show_histo_bgr_mask(show_color_gorilla)


# %%
hsv_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2HSV)


# %%
hsv_gorilla[:,:,2].max()


# %%
hsv_gorilla[:,:,2].min()


# %%
show_histo_bgr_mask(hsv_gorilla)


# %%
hsv_gorilla[:, :, 2] = cv2.equalizeHist(src=hsv_gorilla[:, :, 2])


# %%
eq_color_gorilla = cv2.cvtColor(src=hsv_gorilla, code=cv2.COLOR_HSV2RGB)


# %%
display_img(eq_color_gorilla)


# %%
show_histo_bgr_mask(eq_color_gorilla)


# %%



