import torch
import torchvision.transforms as transforms
from PIL import Image
from inferrence import grady_search

device =torch.device('cuda' if torch.cuda.is_available else 'cpu')
def print_examples(encoder, decoder, vocabulary, beam_size=3):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    decoder.eval()
    test_img1 = "test_examples/cat.jpg"
    test_img1_predict=grady_search(encoder, decoder, test_img1, vocabulary)
    print(
        "Example 1 OUTPUT: "
        + " ".join(test_img1_predict)
    )
    test_img2 = "test_examples/dog.jpg"
    test_img2_predict=grady_search(encoder, decoder, test_img2, vocabulary)
    print(
        "Example 2 OUTPUT: "
        + " ".join(test_img2_predict)
    )
    test_img3 = "test_examples/tree.jpg"
    test_img3_predict=grady_search(encoder, decoder, test_img3, vocabulary)
    print(
        "Example 3 OUTPUT: "
        + " ".join(test_img3_predict)
    )
    test_img4 = "test_examples/mountains.jpg"
    test_img4_predict=grady_search(encoder, decoder, test_img4, vocabulary)
    print(
        "Example 4 OUTPUT: "
        + " ".join(test_img4_predict)
    )
    test_img5 = "test_examples/happy.jpg"
    test_img5_predict=grady_search(encoder, decoder, test_img5, vocabulary)
    print(
        "Example 5 OUTPUT: "
        + " ".join(test_img5_predict)
    )
    decoder.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step