import os
import json

class CustomDataset:
    def __init__(self, path):
        self.json_files = [path+f for f in os.listdir(path) if f.endswith('.json')]
        self.arr = []
        for f in self.json_files:
            with open(f, 'r', encoding='utf-8') as file:
                d = json.load(file)
                if 'text' in d.keys():
                    img_path = f.split('.json')[0].split("/")[-1]
                    for i in d['text']:
                        self.arr.append((img_path+".jpg", i, img_path+"_mask.png"))
                else:
                    l = len(d.keys()) - 1
                    for i in range(l):
                        self.arr.append((d['img'], d[str(i)], d['img'].split(".")[0]+"_mask.png"))
                file.close()
    def __getitem__(self, index):
        return self.arr[index]
    def __len__(self):
        return len(self.arr)


if __name__ == "__main__":
    ds = CustomDataset("../../cot-seg_code/datasets/ReasonSeg-Hard/")
    for i in range(len(ds)):
        print(ds[i])



