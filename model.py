import torch
import torch.nn as nn
import torchvision.models as models
from get_loader import get_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, encode_image_size=14, fine_tune=False):
        super(Encoder, self).__init__()
        resnet50 = models.resnet50(pretrained=True)# aux_logits: special parameters of inception
        #Remove linear and pool layers
        modules = list(resnet50.children())[:-2]
        self.resnet50 = nn.Sequential(*modules)

        # AdaptiveAvgPool2d-->自适应池化层，将输入尺寸变换为 encode_image_size * encode_image_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encode_image_size,encode_image_size))

        # 是否训练resnet50
        self.fine_tune(fine_tune)

    def forward(self,images):

        out = self.resnet50(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=False):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet50.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet50.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_size, attention_dim):
        """
        :param feature_dim: feature channels of encoded images
        :param hidden_size: hidden size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(feature_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(hidden_size, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """  #encoder_out (2,196,2048), decoder_hidden(2,512)
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)(2,196,512)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)(2,512)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels) (2,196)
        alpha = self.softmax(att)  # (batch_size, num_pixels) (2,196)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim) (2,2048)

        return attention_weighted_encoding, alpha

class DecoderWithAttention(nn.Module):

    def __init__(self, attention_dim, embed_dim, hidden_size, vocab_size, feature_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param hidden_size: hidden size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param feature_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(feature_dim, hidden_size, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + feature_dim, hidden_size, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(feature_dim, hidden_size)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(feature_dim, hidden_size)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(hidden_size, feature_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        #encoder_out (2,14,14,2048), encoded_captions (2,20)
        batch_size = encoder_out.size(0)
        feature_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size # 10000

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, feature_dim)  # (batch_size, num_pixels, encoder_dim) (2,196,2048)
        num_pixels = encoder_out.size(1) #14*14=196

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim) [2,20,512]

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim) (2,512)

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, encoded_captions.shape[1], vocab_size).to(device) #shape [2,20,10000]
        alphas = torch.zeros(batch_size, encoded_captions.shape[1], num_pixels).to(device) #shape [2,20,196]

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(encoded_captions.shape[1]):
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim) （2，2048）
            attention_weighted_encoding = gate * attention_weighted_encoding #（2，2048）
            h, c = self.decode_step(torch.cat([embeddings[:, t, :], attention_weighted_encoding], dim=1),(h, c))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size) （2，10000）
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha

        return predictions, encoded_captions,alphas


if __name__=='__main__':
    vocab_size = 10000
    emb_dim = 512  # dimension of word embeddings
    attention_dim = 512  # dimension of attention linear layers
    hidden_size = 512  # dimension of decoder RNN, 其实就是hidden_size
    dropout = 0.5
    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       hidden_size=hidden_size,
                                       vocab_size=vocab_size,
                                     dropout=dropout)
    encoder = Encoder()
    decoder = decoder.to('cuda')
    encoder = encoder.to('cuda')
    img = torch.ones(2,3,448,448).to('cuda')
    cap = torch.ones(2,20,dtype=int).to('cuda')
    img = encoder(img) #(1,3,448,448)--->(2,14,14,2048)
    scores, caps_sorted, alphas = decoder(img, cap)
