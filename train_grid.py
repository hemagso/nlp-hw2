from hw2.data import load_dataset
from hw2.models.sequence import EncoderDecoder
from torch.utils import data
from hw2.utils import train, save, create_grid, build_model_folder
import torch
import os


def train_grid(params):
    train_ds, test_ds = load_dataset("./data", sampling=params["SAMPLING"])

    print_every = len(train_ds) // (10*params["BATCH_SIZE"])
    print(len(train_ds), print_every)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"  # use gpu whenever you can!

    train_dl = data.DataLoader(train_ds, batch_size=params["BATCH_SIZE"], shuffle=True, num_workers=8)
    test_dl = data.DataLoader(test_ds, batch_size=params["BATCH_SIZE"], shuffle=True, num_workers=8)

    model = EncoderDecoder(
        len(train_ds.src_vocabs),
        len(train_ds.trg_vocabs),
        params["EMBED_SIZE"],
        params["HIDDEN_SIZE"],
        n_layers=params["N_LAYERS"],
        bidirectional=params["BIDIRECTIONAL"],
        dropout=params["DROPOUT"]
    ).to(device)

    if not os.path.exists(os.path.join("./models", build_model_folder(model, params))):
        ppl = train(model, params["NUM_EPOCHS"], params["LR"], print_every, train_dl, test_dl, device=device)
        save(model, "./models", params, stats={"perplexity": ppl})
    else:
        print("Skipping already run model", params)


if __name__ == "__main__":
    fixed = {
        "BATCH_SIZE": 128,
        "NUM_EPOCHS": 15,
        "SAMPLING": 0.2,
        "DROPOUT": 0.2
    }
    grid = {
        "EMBED_SIZE": [128, 256, 512],
        "HIDDEN_SIZE": [128, 256, 512],
        "LR": [1E-3, 1E-2],
        "N_LAYERS": [2, 3],
        "BIDIRECTIONAL": [False, True]
    }
    for idx, params in enumerate(create_grid(grid, fixed)):
        print(idx, params)
        if idx in []:
            continue
        else:
            train_grid(params)