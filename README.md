# MRI segmentation
Code used for UW Maddison MRI segmentation [kaggle challenge](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation)

To use code create a folder with *org.py*, *volviz.py* and the relevant zip file in the same folder.  
To convert all files to nifti and create the revelant mask run the following:
```
from org import org
fileorganizer = org()
```

