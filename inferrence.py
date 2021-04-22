import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

import argparse
from PIL import Image
from get_loader import get_loader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
from itos_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Vocabulary():
    def __init__(self,itos,stoi):
        self.itos = itos
        self.stoi = stoi

def caption_image_beam_search(encoder, decoder, image_path, vocabulary, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param vocabulary: word vocabulary
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(vocabulary.itos)

    # Read image and process
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0).to(device)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[vocabulary.stoi['<SOS>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = (top_k_words / vocab_size)  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds.long()], alpha[prev_word_inds.long()].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != vocabulary.stoi['<EOS>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds].long()]
        c = c[prev_word_inds[incomplete_inds].long()]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return [vocabulary.itos[idx] for idx in seq], alphas

def grady_search(encoder, decoder, image_path, vocabulary, max_length=30):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0).to(device)  # (1, 3, 256, 256)
    result_caption=[]
    with torch.no_grad():
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.shape[1]
        encoder_out = encoder_out.view(1, -1, encoder_out.shape[3])
        #initializing the hidden and cell
        h, c = decoder.init_hidden_state(encoder_out)
        x = torch.tensor([vocabulary.stoi['<SOS>']]).unsqueeze(0).to(device) #(1,1),the first word
        for _ in range(max_length):
            embeddings = decoder.embedding(x).squeeze(dim=1)  # (1,1,embedding_size)

            awe, alpha = decoder.attention(encoder_out, h)

            alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = scores.argmax(1)
            x = scores #output gonna be the next input
            result_caption.append(scores.item())
            if vocabulary.itos[scores.item()]=="<EOS>": #最长长度设置为50，遇到EOS就停止
                break
    return [vocabulary.itos[idx] for idx in result_caption]

def visualize_att(image_path, seq, alphas, smooth=True, plot=True):
    """
    Visualizes caption with weights at every word.
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    for t in range(len(seq)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(seq) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (seq[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t]
        if smooth:
            alpha = skimage.transform.pyramid_expand(np.array(current_alpha), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(np.array(current_alpha), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.savefig('attention_map.jpg')
    if plot:
        plt.show()


def inferrence(encoder,decoder,vocab,image_path,plot=True):
    predict,alphas = caption_image_beam_search(encoder, decoder, image_path, vocab)
    visualize_att(image_path, predict, alphas, smooth=False,plot=plot)
    return predict

if __name__=="__main__":
    #load vocabulary
    vocab_itos = load_vocab('vocabulary/itos.txt')
    vocab_stoi = load_vocab('vocabulary/stoi.txt')
    vocab = Vocabulary(vocab_itos,vocab_stoi)
    print('Vocabulary loaded')

    #load model
    print('loading model')
    checkpoint = torch.load('checkpoint/my_checkpoint.pth.tar')
    encoder = checkpoint['encoder'].to(device)
    decoder = checkpoint['decoder'].to(device)
    encoder.eval()
    decoder.eval()
    print('model loaded')

    #image path and plot
    image_path = 'test_examples/mountains.jpg'
    plot = True # plot attention map or not
    #predict
    print('Predicting')
    #predict = grady_search(encoder, decoder, image_path, dataset.vocab, max_length=30)
    predict = inferrence(encoder,decoder,vocab,image_path,plot=plot)
    predict = predict[1:-1] #filter the <sos> and <eos>
    print('PREDICT: ',' '.join(predict))

