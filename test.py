import torch
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    model.eval()


    test_loss, test_acc = 0,0

    with torch.inference_mode():
        for batch, (X,y)in enumerate(dataloader):
            X,y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits,y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/ len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len (dataloader)
    
    return test_loss,test_acc