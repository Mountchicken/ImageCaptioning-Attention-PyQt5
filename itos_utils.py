from get_loader import get_loader
#保存itos字典
def save_vocab(itos,save_path):
    with open(save_path,'w') as f:
        f.write(str(itos))

def load_vocab(save_path):
    with open(save_path,'r') as f:
        dic=f.read()
    return eval(dic)

if __name__=="__main__":
    _, dataset=get_loader(
        root_folder="flickr30k/flickr30k-Images",
        annotation_file="flickr30k/flickr30k/captions.txt",
        transform=None,
        batch_size=64,
        num_workers=0
    )
    save_vocab(dataset.vocab.itos,'vocabulary/itos.txt')
    save_vocab(dataset.vocab.stoi,'vocabulary/stoi.txt')