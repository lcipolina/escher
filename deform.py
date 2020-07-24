import numpy as np
import cv2
from PIL import Image


deformation_scale = 256
droste_scale = 256

# http://www.ams.org/notices/200304/fea-escher.pdf
# code based on:
# https://www.shadertoy.com/view/Mdf3zM

def escher_deformation(x, y):
    # convert cartesian to polar
    # float lnr = log(length(uv));
    lnr = np.log((x**2 + y**2)**0.5)
    # float th = atan( uv.y, uv.x )+(0.4/256.)*deformationScale;
    th = np.arctan2(y, x) + (0.4 / 256) * deformation_scale
    # float sn = -log(deformationScale)*(1./(2.*3.1415926));
    sn = -np.log(deformation_scale) * (1 / (2 * np.pi))
    # float l = exp( lnr - th*sn );
    l = np.exp(lnr - th * sn)

    # vec2 ret = vec2( l );
    ret_x, ret_y = l.copy(), l.copy()

    # ret.x *= cos( sn*lnr+th );
    ret_x *= np.cos(sn * lnr + th)
    # ret.y *= sin( sn*lnr+th );
    ret_y *= np.sin(sn * lnr + th)

    return ret_x, ret_y


def droste_transformation(x, y):
    for _ in range(2):
        # if(any(greaterThan(abs(uv),vec2(1.)))):
        indx = (np.abs(x) > 1) | (np.abs(y) > 1)
        # uv *= (1./drostescale);
        x[indx] *= 1. / droste_scale
        y[indx] *= 1. / droste_scale

        # if(all(lessThan(abs(uv),vec2(1./drostescale)))):
        indx = (np.abs(x) < 1 / droste_scale) & (np.abs(y) < 1 / droste_scale)
        # uv *= drostescale;
        x[indx] *= droste_scale
        y[indx] *= droste_scale
    return x, y


# m is the size of the src image
m = 600
src = np.array(Image.open('images/escher_straight.jpg'))
src = np.stack([src[:444, :]] * 3, axis=2)
m = src.shape[0]

# n is the size of the output image
n = 500
x, y = np.meshgrid(np.linspace(-1, 1, n, dtype=np.float32),
                   np.linspace(-1, 1, n, dtype=np.float32))
x, y = escher_deformation(x, y)
x, y = droste_transformation(x, y)
x = (x + 1) * ((m - 1) / 2)
y = (y + 1) * ((m - 1) / 2)

# out = cv2.remap(src, x, y, interpolation=cv2.INTER_NEAREST)
out = cv2.remap(src, x, y, interpolation=cv2.INTER_CUBIC)

src_img = Image.fromarray(src)
out_img = Image.fromarray(out)

out_img.save('images/deform_attempt.png')
