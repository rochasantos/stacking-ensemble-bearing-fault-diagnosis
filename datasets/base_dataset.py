import os
import numpy as np
from src.utils import download_file, extract_rar
from src.data_processing.data_manager import DatasetManager


from abc import ABC, abstractmethod

class BaseDataset(ABC):
    
    def __init__(self, rawfilesdir, url):
        """
        Base class for all dataset models. 
        Defines the attributes and the download and load_signal functions, 
        along with an abstract extract_data method. The extract_data method 
        delegates the responsibility of data extraction to the subclasses, 
        requiring them to implement their specific extraction logic.

        Parameters:
        - rawfilesdir (str): The directory where raw files will be stored.
        - url (str): The base URL for downloading the dataset files.
        
        Methods:
            download(): Downloads .mat from the dataset website URL.
        """
        self._rawfilesdir = rawfilesdir  # Directory to store the raw files
        self._url = url  # Base URL for downloading the files
        self.acquisition_maxsize = None  # Maximum size for data acquisition.
        self._metainfo = DatasetManager(str(self))
        if not os.path.exists(self._rawfilesdir):
            os.makedirs(self._rawfilesdir)

    def download(self):
        """ Download files from datasets website.
        """
        url = self._url
        dirname = self.rawfilesdir
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        print(f"Stating download of {self} dataset.")
        list_of_bearings = self.list_of_bearings()
        dataset_name = self.__class__.__name__.lower()
        unit = '.mat'
        # if dataset_name == "paderborn":
        #     unit = '.rar'
        # if dataset_name == "muet":
        #     unit = '.csv'
        for bearing in list_of_bearings:
            sufix_url = bearing[1]
            output_path = os.path.join('data/raw', dataset_name, bearing[0]+unit)
            if not os.path.exists(os.path.join(dirname, sufix_url)):
                download_file(url, sufix_url, output_path)                
            # if unit == '.rar':
            #     extract_rar(output_path, output_path[:-4])
        print("Download finished.")


    def process_and_save_signal(self, output_root=None, filter=None, max_size=None, pipeline_transforms=None):
        
        dataset_name = self.__class__.__name__.lower()
        
        if output_root is None:
            output_root = f"data/processed/{dataset_name}" # Adjust as needed on case CWRU severity /007, /014 or /021
        
        if filter:
            if isinstance(filter, list): # paderborn case
                print(self._metainfo.filter_data()[0]["filename"])
                metainfo = [info for info in self._metainfo.filter_data() if info["filename"].split("_")[3] in filter]
            if isinstance(filter, dict):
                print(f"Filter: {filter} is dict")
                metainfo = self._metainfo.filter_data(filter)
        else:
            metainfo = self._metainfo.filter_data()

        print(f"Metainfo: {len(metainfo)}")
        segments_counter = 0  
        for info in metainfo:
            basename = info["filename"]
            original_fs = int(info["sampling_rate"])
            label = info["label"]
            
            output_dir = os.path.join(output_root, label)
            os.makedirs(output_dir, exist_ok=True)

            filepath = os.path.join('data/raw/', self.__class__.__name__.lower(), basename+'.mat')
            if dataset_name == "paderborn":
                subdir = basename.split("_")[3]
                filepath = os.path.join('data/raw/', self.__class__.__name__.lower(), subdir, subdir, basename)  
            signal, label = self._extract_data(filepath)   

            if max_size and len(signal) > max_size:
                signal = signal[:max_size]
            if max_size and len(signal) < max_size:
                print(f"Signal length {len(signal)} in the filename {basename}.mat is less than max size {max_size}. Skipping.")
                continue
            if pipeline_transforms:
                signal = pipeline_transforms.apply(signal, original_fs)
            np.save(os.path.join(output_dir, f"{basename}.npy"), signal)
            
            # if segment_size:                
            #     for segment in range(0, len(signal), segment_size):
            #         segment_signal = signal[segment:segment + segment_size]
            #         if len(segment_signal) < segment_size:
            #             continue
            #         if pipeline_transforms:
            #             segment_signal = pipeline_transforms.apply(segment_signal, original_fs)
            #         # Save the processed signal
            #         segments_counter += 1
            #         np.save(os.path.join(output_dir, f"{basename}_{segment//segment_size}.npy"), segment_signal)

        print(f"Processed {segments_counter} segments from {len(metainfo)} files.")

    
    def load_file(self, filepath):
        signal, label = self._extract_data(filepath)
        return signal, label
        

    def load_data(self, filter=None):                    
        if filter:
            if isinstance(filter, list): # paderborn case
                print(self._metainfo.filter_data()[0]["filename"])
                metainfo = [info for info in self._metainfo.filter_data() if info["filename"].split("_")[3] in filter]
            if isinstance(filter, dict):
                metainfo = self._metainfo.filter_data(filter)
        else:
            metainfo = self._metainfo.filter_data()

        dataset_name = self.__class__.__name__.lower()
        for info in metainfo:
            basename = info["filename"]
            filepath = os.path.join('data/raw/', dataset_name, basename+'.mat')
            # if dataset_name == "cwru":
            #     if info["hp"] == "0": # skip 0 HP
            #         continue
            if dataset_name == "paderborn":
                rotation_label = basename.split("_")[0]
                if rotation_label != "N15": # skip N15
                    continue
                if basename == "N15_M07_F10_KA01_8":
                    continue
                subdir = basename.split("_")[3]
                filepath = os.path.join('data/raw/', dataset_name, subdir, subdir, basename)
            signal, label = self._extract_data(filepath)
            yield signal, label, int(info["sampling_rate"]), basename

    def load_data_f(self, filter=None):                    
        if filter:
            if isinstance(filter, list): # paderborn case
                print(self._metainfo.filter_data()[0]["filename"])
                metainfo = [info for info in self._metainfo.filter_data() if info["filename"].split("_")[3] in filter]
            if isinstance(filter, dict):
                metainfo = self._metainfo.filter_data(filter)
        else:
            metainfo = self._metainfo.filter_data()

        dataset_name = self.__class__.__name__.lower()
        for info in metainfo:
            basename = info["filename"]
            extent_damage = info["extent_damage"]
            load = info["hp"]
            filepath = os.path.join('data/raw/', dataset_name, basename+'.mat')
            # if dataset_name == "cwru":
            #     if info["hp"] == "0": # skip 0 HP
            #         continue
            if dataset_name == "paderborn":
                rotation_label = basename.split("_")[0]
                if rotation_label != "N15": # skip N15
                    continue
                if basename == "N15_M07_F10_KA01_8":
                    continue
                subdir = basename.split("_")[3]
                filepath = os.path.join('data/raw/', dataset_name, subdir, subdir, basename)
            signal, label = self._extract_data(filepath)
            yield signal, info
            
    def group_by(self, feature, filter=None, sample_size=None, target_sr=42000):
        metainfo = self.get_metainfo(filter)
        groups = []
        hash = dict()
        for i in metainfo:
            ftr = i[feature]
            if ftr not in hash:
                hash[ftr] = len(hash)
            if sample_size:
           