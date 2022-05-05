# MRI segmentation
<sub><sup>WIP</sup></sub>

Code used for UW-Maddison MRI segmentation [kaggle challenge](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation) April 15 - July 8

The coe below utilitzes multiprocessing, the default number of cores is set to 12. If you want to use  ```N``` number of cores run: 
``` 
from org import org
fileorganizer = org()
fileorganizer.cores = N
```


To unzip and organize the images and convert to nifti run the following:
````
from org import org
fileorganizer = org()
fileorganizer.unzip()
fileorganizer.scans2nifti()
````
Note it will change the slicing axis to the standard format. Before running the code the k'th slice would be accesed as ```img[k,:,:]``` while the standard way is ```img[:,:,k]```. ````scans2nifti```` will convert so the slice axis is the standard scans2nifti.
The relevant zip file should be in the same folder as *org.py*

To generate the segmentation mask from the ```train.csv``` file run
````
from org import org
fileorganizer = org()
fileorganizer.mask_all_volumes()
````
Note that this is somewhat time consuming. It is asumed that ```.scans2nifti() ``` has been run prior to generating the segmentation masks.

Vizualizing a volume is done using a tuple of the patient and the given day `(<case>,<case_day>)`.
`````
from org import org
fileorganizer = org()
pair = ("case123", "case123_day0")
fileorganizer.volviz(pair)
`````