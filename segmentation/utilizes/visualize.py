import os
import skimage
import cv2
import nibabel as nib
import tqdm


def save_img(path, img, lib="cv2", overwrite=True):
    if not overwrite and os.path.exists(path):
        pass
    else:
        print(path)
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if lib == "skimage":
            skimage.io.imsave(path, img)
        elif lib == "cv2":
            cv2.imwrite(path, img)
        elif lib == "nib":
            nib.save(nib.Nifti1Image(img, None), path)


