import os
import pandas as pd
import spacy
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

spacy_eng = spacy.load("en_core_web_sm")
class Vocabulary:
    def __init__(self, freq_threshold): #freq_threshold:只有出现次数大于freq_threshold的词，我们才把他纳入词典中
        self.itos = {0:"<UNK>", 1:"<SOS>", 2:"<EOS>", 3:"<PAD>"} #index to string
        self.stoi = {"<UNK>":0, "<SOS>":1, "<EOS>":2, "<PAD>":3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies={}
        idx =4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self,text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class Flickr30kDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        '''
        Flickr30K
        1.总共有31783张图片，每张图片有5句描述，总共31783*5=158915句描述
        2.五句话按照句子长短排列，句子越长描述越详细，选择每五句话的第三句进行训练
        '''
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file,sep = '\t',skiprows=lambda x: x > 0 and (x+2)%5 != 0)
        self.transform = transform
        #Get img, caption columms
        self.imgs = self.df['image']
        self.captions = self.df['caption']
        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist()) #send in all the captions to build a vocabulary

    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        caption = self.captions[index] #选择每五句话的第三句进行训练
        img_id = self.imgs[index][:-2] #去除 #3
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Build the numericalized captions
        numericalized_caption = [self.vocab.stoi["<SOS>"]] #The Start sign
        numericalized_caption += self.vocab.numericalize(caption) # Convert the cation into numbers
        numericalized_caption.append(self.vocab.stoi["<EOS>"]) #The End sign

        return img, torch.tensor(numericalized_caption)

class MyCollate:
    def __init__(self,pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0)for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return imgs, targets

def get_loader(
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=0,
    shuffle=True,
    pin_memory=True,
):
    dataset = Flickr30kDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )
    return loader, dataset
if __name__=="__main__":
    transforms = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ]
    )
    dataloader,dataset=get_loader("flickr30k-images/",annotation_file='flickr30k/captions.txt',transform=transforms)
    count=0
    for idx, (imgs,captions)in enumerate(dataloader):
        if count!=30:
            print(imgs.shape)
            print(captions.shape)
            count+=1
        else:
            break
    print(len(dataset.vocab))