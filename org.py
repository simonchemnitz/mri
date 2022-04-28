import enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nibabel as nib
import multiprocessing as mp

import zipfile
import os
import glob

from volviz import plotly_volume

class org:
    def __init__(self):
        self.wd = os.getcwd()#Current working directory
        self.cores = 12 #Number of cores to use when multiprocessing
        self.data_dir = "" #Directory where data is stored
        self.nifiti_dir = "" #Directory where nifti files are stored
        self.zipname = "" #Name of zip file
        self.patients = [] #List of patients
        self.pairs = [] #List of pairs (patient,day)
        self.segpairs = [] #List of pairs (patient,day) where segmentation nonzero
        self.maskdf = pd.DataFrame() #Dataframe with mask information
        self.segval = {} #Dictionary with segmentation class as key and unique int as value
        ####################
        ###   Functions  ###
        ####################
        print()
        print("Making directories:")
        self.makedir()#Make the data directory
        print()
        print("Unzipping files:")
        self.unzip()#Unzip the files

        #Patients
        self.init_patients()
        
        #Nifti conversion
        self.scans2nifti()

        #Mask DataFram
        self.maskdf = self.create_maskdf()

    def makedir(self):
        """
        Make data directory if they do not exist
        and initialise data and nifti dir
        """

        #Data Dir
        if not os.path.exists(self.wd + '/data'):
            print(self.wd+"/data/")
            os.mkdir(self.wd+"/data/")
        
            
        #Nifti Dir
        if not os.path.exists(self.wd + '/data/nifti'):
            print(self.wd+"/data/nifti/")
            os.mkdir(self.wd+"/data/nifti/")

        #Initialize data and nifti dir
        self.data_dir = self.wd + '/data/'
        self.nifiti_dir = self.wd + '/data/nifti/'

    def unzip(self):
        """
        Unzip data files into data folder
        """
        #Check if already unzipped
        condition = os.path.exists(self.data_dir + 'raw')
        if not (condition):
            #Find zipfile 
            zip_files1 = glob.glob(self.wd + '/data/*.zip')
            zip_files2 = glob.glob(self.wd + '/*.zip')
            zip = zip_files1 + zip_files2

            #unzip file(s)
            for zf in zip:
                self.zipname = os.path.basename(zf)
                print()
                print("Unzipping file: " + self.zipname)
                print("at location: ", self.data_dir)
                with zipfile.ZipFile(zf, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                print("Unzipping complete")
                print("Renaming folder from <train> to <raw>")
                os.rename(self.data_dir + 'train', self.data_dir + 'raw')
                print()
        else:
            print("File already unzipped")
            print()
        return None

    def init_patients(self):
        """
        Initialize self.patients and self.pairs
        self.patient is a list with all patients
        self.pairs is a list with all (patient,day) pairs
        """
        #initialize patients
        self.patients = sorted(os.listdir(self.data_dir + 'raw/'))
        #initialize pairs
        self.pairs = []
        for patient in self.patients:
            days = sorted(os.listdir(self.data_dir + 'raw/' + patient + '/'))
            for day in days:
                self.pairs.append((patient, day))

    def reshape_img(self, arr):
        """
        Reshape image such that slicing is standard.
        I.e. slicing axis is the second axis
        eg img[:,:,k] is the k'th slice.

        Parameters
        ----------
        arr : numpy array
            The image to be reshaped

        Returns
        -------
        numpy array
            The reshaped image
        """
        arr = np.swapaxes(arr,0,2)
        arr = np.swapaxes(arr,0,1)
        return arr


    def scans2nifti_1(self, pair):
        """
        Convert one (1) scans folder (image volume) to nifti .nii format
        """
        patient, day = pair
        scan_folder = self.data_dir + 'raw/' + patient + '/' + day + '/scans/'
        files = sorted(os.listdir(scan_folder))
        #Load images to numpy array
        arr = np.array([plt.imread(scan_folder+file) for file in files])
        img = self.reshape_img(arr)
        
        #Save to nifti format
        #print(self.nifiti_dir)
        #print(patient)
        #print(day)
        #print()
        #Check if directory exists
        if not os.path.exists(self.nifiti_dir + patient):
            #print("Making directory: ", self.nifiti_dir + patient)
            os.mkdir(self.nifiti_dir + patient)
        #print("Filename will be:")
        filename = self.nifiti_dir + patient + '/' + day + '.nii'
        #print(filename)
        if not os.path.exists(filename):
            #save as nifti
            nib.save(nib.Nifti1Image(img, np.eye(4)), filename)
            
            #print("loading to check if correct")
            #load nifti and check if correct
            #img_nifti = nib.load(filename)
            #img_nifti_arr = img_nifti.get_fdata()
            #print( (img_nifti_arr[:,:,0]==arr[0,:,:]).all() )
        return None

    def scans2nifti(self):
        """
        Convert all scans to nifti
        """
        #Check if nifti folder is empty
        if not os.listdir(self.nifiti_dir):
            #Convert to nifti in parallel
            print()
            print("Converting scans to nifti in parallel")
            print("Using", self.cores, "cores")
            with mp.Pool(processes=self.cores) as pool:
                results = pool.map(self.scans2nifti_1, self.pairs)
                pool.close()
            results = None
            print("Conversion complete")
        else:
            print()
            print("Nifti folder not empty")
            print("Skipping conversion")
            print("To check if conversion was successful run")
            print("self.mass_nifti_check")

    def nifti_conversion_check(self, pair):
        """
        Check if convertion to nifti is done correctly
        return true if
        """
        patient, day = pair
        #Load regular file
        scanfolder = self.data_dir + 'raw/' + patient + '/' + day + '/scans/'
        files = sorted(os.listdir(scanfolder))
        arr = np.array([plt.imread(scanfolder+file) for file in files])
        #load nifti file
        img_nifti = nib.load(self.nifiti_dir + patient + '/' + day + '.nii')
        img_nifti_arr = img_nifti.get_fdata()
        #check if same img
        return ( (img_nifti_arr[:,:,0]==arr[0,:,:]).all() )

    def mass_nifti_check(self):
        """
        Check if all nifti files are converted correctly
        by comparing the first slice of the relevant volume image
        """
        #in paralell check if all nifti files are converted correctly
        print("Checking if all nifti files are converted correctly")
        print("Using", self.cores, "cores")
        with mp.Pool(processes=self.cores) as pool:
            results = pool.map(self.nifti_conversion_check, self.pairs)
            pool.close()
        print("Checking complete")
        print("Number of nifti files converted correctly: ", sum(results))
        print("Number of nifti files converted incorrectly: ", len(results)-sum(results))

    def volviz(self, pair):
        """
        Visualize one volume image with plotly
        given a (patient, day) pair
        """
        patient, day = pair
        #Load relevant nifti file
        img_nifti = nib.load(self.nifiti_dir + patient + '/' + day + '.nii')
        #Convert to numpy array
        img_nifti_arr = img_nifti.get_fdata()
        #Plot volume image
        fig = plotly_volume(img_nifti_arr)
        return None

    def sliceviz(self, pair, slice):
        """
        Visualize one slice of a volume image
        """
        patient, day = pair
        #Load relevant nifti file
        img_nifti = nib.load(self.nifiti_dir + patient + '/' + day + '.nii')
        #Convert to numpy array
        img_nifti_arr = img_nifti.get_fdata()
        #Plot volume image
        fig = plt.figure()
        plt.imshow(img_nifti_arr[:,:,slice], cmap = "gray")
        plt.axis("off")
        plt.title("Slice " + str(slice) + " of " + patient + " " + day)
        plt.show()
        return None

    def create_maskdf(self):
        """
        Creates a dataframe with columns:
        patient, day, mask_name, mask_value
        in addition to already existing columns

        Returns
        -------
        self.maskdf : pandas DataFrame
        """
        #Load csv file
        maskdf = pd.read_csv(self.data_dir + '/train.csv')
        #Add new columns
        maskdf["patient"] = maskdf["id"].apply(lambda x: x.split("_")[0])
        maskdf["day"] = maskdf["patient"] + "_" + maskdf["id"].apply(lambda x: x.split("_")[1])
        maskdf["slice"] = maskdf["id"].apply(lambda x: int(x.split("_")[-1]))
        #Set maskdf
        self.maskdf = maskdf
        self.maskdf = self.maskdf.dropna(subset=["segmentation"])
        self.maskdf = self.maskdf.reset_index(drop=True)

        #set pairs of patients who have nonzero masks
        days = self.maskdf["day"].unique()
        self.segpairs = []
        for day in days:
            patient = day.split("_")[0]
            self.segpairs.append((patient, day))
        
        segclasses = self.maskdf["class"].unique()
        self.segval = {segclass:i for i,segclass in enumerate(segclasses)}
        return self.maskdf

    def mask_one_volume(self, pair):
        """
        Given a pair of (patient, day) create a relevant
        segmentation mask
        """
        patient, day = pair
        #Load relevant nifti file
        img_nifti = nib.load(self.nifiti_dir + patient + '/' + day + '.nii')
        #Convert to numpy array
        img_nifti_arr = img_nifti.get_fdata()
        #dimension of volume
        dim = img_nifti_arr.shape
        #Create empty mask
        mask = np.zeros(dim)

        #Relevant rows of maskdf
        maskdf_sub = self.maskdf[(self.maskdf["patient"]==patient) & (self.maskdf["day"]==day)]
        
        #Loop through rows and create mask
        for index, row in maskdf_sub.iterrows():
            #Get relevant slice
            slice = row["slice"]
            seg_string = row["segmentation"]
            seg_class = row["class"]
            print(slice)
            print(seg_string)
            print(seg_class)
            print()
            print()
            #Get relevant mask
        return maskdf_sub
    def parse_string(self, string, dim):
        """
        Given a segmentation string
        create the relevant binary mask
        """
        #Create flattend empty mask
        mask = np.zeros(dim[0]*dim[1])
    def parse_segmentation_string(self, string, pair):
        """
        Parse the segmentation string in self.maskdf
        """
        #Load relevant nifti file
        img_nifti = nib.load(self.nifiti_dir + pair[0] + '/' + pair[1] + '.nii')

        #Convert to numpy array
        img_nifti_arr = img_nifti.get_fdata()

        #first slice
        fslice = img_nifti_arr[:,:,0]
        img_length = len(fslice.flatten())
        print(img_length)
        mask = np.zeros(img_length)

        #Split string
        string_list = string.split(" ")
        pixel_loc = string_list[::2]
        pixel_length = string_list[1::2]
        print(pixel_loc)
        print(pixel_length)
        for location, length in zip(pixel_loc, pixel_length):
            print(location)
            print(length)
            mask[int(location):int(location)+int(length)] = 1
        return mask.reshape(img_nifti_arr.shape[0], img_nifti_arr.shape[1])

    def load_nifti_img(self, pair):
        """
        Given a (patient, day) pair return numpy array of relevant image
        """
        #Load relevant nifti file
        img_nifti = nib.load(self.nifiti_dir + pair[0] + '/' + pair[1] + '.nii')
        #Convert to numpy array
        img_nifti_arr = img_nifti.get_fdata()
        return img_nifti_arr