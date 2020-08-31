import pandas as pd
import ujson as json
import collections
import os


class Extr_json:
    def __init__(self, path, files = {"dev.jsonl":500, 
                                    "train.jsonl": 8500, 
                                    "test.jsonl": 1000}):
        self.path = path
        self.files = files

    def __extract_features_csv(self, record, type = 'train'):
        row = [
                ('id', record['id']),
                ('img', record['img']),
                ('text', record['text'])
            ]
        if type == 'train':
            row.append(
                ('label', record['label'])
            )
        return collections.OrderedDict(row)

    def __read_data(self, file):
        with open(file) as fin:
            for line in fin:
                yield json.loads(line) 
    
    def extract(self):
        df = []
        for file in self.files.keys():
            data = []
            for record in self.__read_data(self.path/file):
                if file == "test.jsonl":
                    features = self.__extract_features_csv(record, type = 'test')
                else:
                    features = self.__extract_features_csv(record)
                data.append(features)

            dev = pd.DataFrame.from_records(data).set_index('id')
            df.append(dev)
        return df