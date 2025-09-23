import csv
import yaml

class DatasetManager:
    def __init__(self, dataset_name=None):
        self.annotation_file = self._load_annotation_file()
        self.dataset_name=dataset_name

    def _load_annotation_file(self):
        data = []
        with open("data/annotation_file.csv", mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data

    def _load_config(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)   

    def filter_data(self, filter_config=None):        
        
        if self.dataset_name:
            params = filter_config or {}
            filter_config = {"dataset_name": self.dataset_name, **params}

        if not filter_config:
            return self.annotation_file                  
        
        filtered_data = []
        for item in self.annotation_file:
            matches = all(
                item.get(key) in value if isinstance(value, list) else item.get(key) == value
                for key, value in filter_config.items()
            )
            if matches:
                filtered_data.append(item)
        
        return filtered_data

    def get_info(self):
        return self.filter_data()