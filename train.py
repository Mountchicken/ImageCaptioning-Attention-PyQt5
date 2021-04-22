import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import load_checkpoint, save_checkpoint, print_examples
from get_loader import get_loader
from model import Encoder, DecoderWithAttention
from tqdm import tqdm

def train():
    transform = transforms.Compose(
        [
            transforms.Resize((270,270)),
            transforms.RandomCrop((256,256)), #the input size of inception network
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ]
    )
    train_loader, dataset=get_loader(
        root_folder="flickr30k-images",
        annotation_file="flickr30k/captions.txt",
        transform=transform,
        batch_size=64,
        num_workers=2,
        shuffle=True
    )
    #Set some hyperparamters
    torch.backends.cudnn.benchmark = True #Speed up the training process
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    load_model = False
    checkpoint_path = 'checkpoint/my_checkpoint.pth.tar'
    save_model = True
    fine_tune_encoder = False
    emb_dim = 512  # dimension of word embeddings
    attention_dim = 512  # dimension of attention linear layers
    hidden_size = 512  # dimension of decoder RNN
    dropout = 0.5
    vocab_size = len(dataset.vocab)
    learning_rate = 1e-4
    num_epochs = 120
    grad_clip = 5.
    #for tensorboard
    writer = SummaryWriter("runs/flickr")
    step=0
    #initialize model, loss etc

    if load_model is False:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       hidden_size=hidden_size,
                                       vocab_size=vocab_size,
                                       dropout=dropout).to(device)
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=learning_rate)
        encoder = Encoder().to(device)
    else:
        checkpoint = torch.load(checkpoint_path)
        encoder = checkpoint['encoder']
        decoder = checpoint['decoder']
        optimizer = checkpoint['optimizer']

    criterion=nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])  #对于"<PAD>"的词语不需要计算损失
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, factor=0.1, patience=10, verbose=True)
    decoder.train()
    print('Begins')

    for epoch in range(num_epochs):
        print_examples(encoder, decoder, dataset.vocab)
        if save_model:
            checkpoint={
                "encoder": encoder,
                "decoder" : decoder,
                "optimizer":optimizer,
                "epoch" :step
            }
            save_checkpoint(checkpoint)

        loop = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
        total_loss=0
        count = 0

        for idx, (imgs,captions) in loop:
            count = count + 1
            imgs = imgs.to(device)
            captions = captions.to(device)
            features = encoder(imgs)
            predictions, _ , _= decoder(features, captions[:,:-1]) #EOS标志不需要送进网络训练，我们希望他能自己训练出来
            # outputs :(batch_szie, seq_len, vocabulary_size), 但是交叉熵损失接受二维的tensor
            loss = criterion(predictions.reshape(-1, predictions.shape[2]), captions[:,1:].reshape(-1))
            step+=1
            optimizer.zero_grad()
            loss.backward(loss)
            total_loss+=loss.item()
            #梯度裁剪
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip,grad_clip)
            optimizer.step()
            mean_loss = total_loss/count

        print('{}/{}: mean loss :{}'.format(epoch,num_epochs,mean_loss))
        loop.set_description(f'Epoch[{epoch}/{num_epochs}]')
        loop.set_postfix(mean_loss=mean_loss)
        writer.add_scalar("mean_loss", mean_loss, epoch)
        scheduler.step(mean_loss)

if __name__=="__main__":
    train()