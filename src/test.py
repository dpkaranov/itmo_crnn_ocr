import torch
from torchmetrics import CharErrorRate
import os
from tqdm import tqdm
from src.dataset import CapchaDataset
from src.model import CRNN, init_weights
from src.utils import decode_predictions, vizualize_preds



weights_dir = 'src/weights'
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CRNN()
model.load_state_dict(torch.load(os.path.join(weights_dir,'crnn_model.pth'), map_location=torch.device(device)))
test_ds = CapchaDataset((2, 5))
metric = CharErrorRate()


def test(batch_size = 8):
    test_data_loader = torch.utils.data.DataLoader(test_ds, batch_size = batch_size)
    model.eval()
    model.to(device)
    batch, labels = next(iter(test_data_loader))
    output = model(batch[None,:,:,:].to(device))
    preds = decode_predictions(output)
    # batch = batch.cpu().detach().numpy()
    # vizualize_preds(batch, preds)
    labels = labels.cpu().detach().numpy()
    labs = [''.join([str(int(y)) for y in x if y != 10]) for x in labels]
    print(labs)
    print(preds)
    print('Testing has been completed')
    print('CER - ', metric( preds, labs))



if __name__ == '__main__':
    test()
