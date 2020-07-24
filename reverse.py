import numpy as np
import cv2
from PIL import Image


deformation_scale = 256
droste_scale = 256
some_constant = 0.4


def reverse_escher_deformation(x, y):
    sn_lnr_th = np.arctan2(y, x)
    l = x / np.cos(sn_lnr_th)
    # l = y / np.sin(sn_lnr_th)

    sn = -np.log(deformation_scale) * (1 / (2 * np.pi))

    lnr = (np.log(l) + sn * sn_lnr_th) / (1 + sn**2)
    th = sn_lnr_th - sn * lnr - (some_constant / 256) * deformation_scale

    r = np.exp(lnr)

    ret_x = r * np.cos(th)
    ret_y = r * np.sin(th)

    return ret_x, ret_y


# load high res image of final work
# the indexing at the end trims off the white border
src = np.asarray(Image.open('images/escher_high_res.jpg'))[88:-112, 93:-96, :]
# src = np.array(Image.open('images/deform_attempt.png'))
n = src.shape[0]

# m is the size of the inverted image being created
m = 600
x, y = np.meshgrid(np.linspace(-1, 1, m, dtype=np.float32),
                   np.linspace(-1, 1, m, dtype=np.float32))


"""
Below code needs some work.

A single call to reverse_escher_deformation is not enough to complete the task
at hand. It will leave empty regions in the output image. However, I can make
multiple calls to reverse_escher_deformation with the image at different
droste scales to get the other regions. The below code tries to combine them
together in an intelligent way. I believe my approach to combining them is
flawed. However, there is the bigger problem that the deformation is not
properly removed. You can see this with the `images/reverse_phase_{i}.png`
images. That problem needs to be fixed first.
"""

remap_x = np.full_like(x, np.nan)
remap_y = np.full_like(y, np.nan)
remap_dist = np.zeros_like(x)

for i in range(-2, 2):
    scale = droste_scale**i
    out_x, out_y = reverse_escher_deformation(x * scale, y * scale)
    out_dist = (out_x**2 + out_y**2)**0.5

    more_distant = out_dist > remap_dist
    # why 1 and -1? If I change that I also must change the calculations for
    # x2 and y2 to scale them properly.
    in_range = (out_x > -1) & (out_x < 1) & (out_y > -1) & (out_y < 1)
    index = more_distant & in_range
    remap_x[index] = out_x[index]
    remap_y[index] = out_y[index]
    remap_dist[index] = out_dist[index]
    print(i, index.sum())

    x2 = (out_x + 1) * ((n - 1) / 2)
    y2 = (out_y + 1) * ((n - 1) / 2)
    out = cv2.remap(src, x2, y2, interpolation=cv2.INTER_CUBIC)
    Image.fromarray(out).save(f'images/reverse_phase_{i}.png')


x2 = (remap_x + 1) * ((n - 1) / 2)
y2 = (remap_y + 1) * ((n - 1) / 2)

out = cv2.remap(src, x2, y2, interpolation=cv2.INTER_CUBIC)

src_img = Image.fromarray(src)
out_img = Image.fromarray(out)

out_img.show()
