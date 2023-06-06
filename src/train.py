import torch
from tqdm import tqdm
from src.dataset import CapchaDataset
from src.model import CRNN, init_weights


weights_dir = 'src/weights'


def train():
    epoch = 5
    lr = 0.0000001
    batch_size=32
    train_ds = CapchaDataset((3, 5))
    val_ds = CapchaDataset((2, 5))
    train_data_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
    val_data_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    model = CRNN()
    model.apply(init_weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CTCLoss(blank = 10, zero_infinity = True).to(device)
    losses = []
    model.to(device)
    for x in range(epoch):
        model.train()
        epoch_losses_train = []
        print(f"Epoch {x}")
        for i, data in enumerate(tqdm(train_data_loader)):
            optimizer.zero_grad()
            batch, labels = data
            output = model(batch[None,:,:,:].to(device))
            input_length = torch.full(size = (output.size(1),),
                                      fill_value = output.size(0),
                                     dtype = torch.int32)
            targets_length = torch.full(size = (labels.size(0),),
                                      fill_value = labels.size(1),
                                     dtype = torch.int32)
            loss =  criterion( output, labels.to(device), input_length, targets_length)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            epoch_losses_train.append(loss.item())
        print("Train loss:", np.mean(epoch_losses_train))
        epoch_losses_val = []
        for i, data in enumerate(tqdm(val_data_loader)):
            model.eval()
            batch, labels = data
            output = model(batch[None,:,:,:].to(device))
            input_length = torch.full(size = (output.size(1),),
                                      fill_value = output.size(0),
                                     dtype = torch.int32)
            targets_length = torch.full(size = (labels.size(0),),
                                      fill_value = labels.size(1),
                                     dtype = torch.int32)
            loss =  criterion( output, labels.to(device), input_length, targets_length)
            epoch_losses_val.append(loss.item())
        print("Validation loss:", np.mean(epoch_losses_val))
    save_path = os.path.join(weights_dir, 'crnn_model.pth')
    torch.save(model.state_dict(), save_path)
