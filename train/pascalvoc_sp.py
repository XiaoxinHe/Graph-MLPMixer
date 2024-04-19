import torch
from core.config import cfg, update_cfg
from core.train_helper import run
from core.get_data import create_dataset
from core.get_model import create_model
from sklearn.metrics import f1_score
from tqdm import tqdm


def classification_multi(y_true, y_pred):
    # calculate macro f1 score
    _, pred = torch.max(y_pred, dim=1)
    true = y_true.cpu().numpy()
    pred = pred.cpu().numpy()
    f1 = f1_score(true, pred, average='macro', zero_division=0)
    return f1


def train(train_loader, model, optimizer, evaluator, device, sharp):
    total_loss = 0
    N = 0
    step_performances = []
    criterion = torch.nn.CrossEntropyLoss()
    for data in tqdm(train_loader, desc='Training'):
        if model.use_lap:
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        data = data.to(device)
        optimizer.zero_grad()
        mask = ~torch.isnan(data.y)
        y = data.y.to(torch.long)
        out = model(data)

        loss = criterion(out[mask], y[mask])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs

        # Convert predictions to labels
        train_perf = classification_multi(y_true=y, y_pred=out)
        step_performances.append(train_perf)

    train_loss = total_loss / N
    train_perf = sum(step_performances) / len(step_performances)
    return train_perf, train_loss


@torch.no_grad()
def test(loader, model, evaluator, device):
    total_loss = 0
    N = 0
    y_preds, y_trues = [], []
    criterion = torch.nn.CrossEntropyLoss()
    for data in loader:
        data = data.to(device)
        mask = ~torch.isnan(data.y)
        y = data.y.to(torch.long)
        out = model(data)
        loss = criterion(out[mask], y[mask])
        total_loss += loss.item() * data.num_graphs
        y_preds.append(out)
        y_trues.append(y)
        N += data.num_graphs

    y_trues = torch.cat(y_trues)
    y_preds = torch.cat(y_preds)
    test_perf = classification_multi(y_true=y_trues, y_pred=y_preds)
    test_loss = total_loss/N
    return test_perf, test_loss


if __name__ == '__main__':
    cfg.merge_from_file('train/configs/GraphMLPMixer/pascalvoc_sp.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test, evaluator=None)
