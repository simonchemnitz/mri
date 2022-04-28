# MRI segmentation
Code used for UW-Maddison MRI segmentation [kaggle challenge](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation)

To use code create a folder with *org.py*, *volviz.py* and the relevant zip file in the same folder.  


To unzip and organize the images and convert to nifti run the following:
````
from org import org
fileorganizer = org()
fileorganizer.unzip()
fileorganizer.scans2nifti()
````
Note it will change the slicing axis to the standar way. Before running the code the k'th slice would be accesed as img[k,:,:] while the standard way is img[:,:,k]. ````scans2nifti```` will convert so the slice axis is the standard scans2nifti.