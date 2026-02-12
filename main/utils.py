import torch

def has_valid_bbox(bbox):
    x_min, y_min, x_max, y_max = bbox
    return (
        x_max > x_min and
        y_max > y_min
    )

def train_one_epoch(model, train_loader, optimizer, criterian, bbox_loss, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    count = 0

    for images, disease_id, labels, bboxes, has_bbox in train_loader:
        images = images.to(device)
        disease_id = disease_id.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        bboxes = bboxes.float().to(device)
        has_bbox = has_bbox.to(device)
        
        optimizer.zero_grad()

        # if has_bbox.any():
        #     cls_logits, bbox_preds = model(images, disease_id, return_bbox=True)

        #     mask = has_bbox.unsqueeze(1).float()
        #     loss_bbox = bbox_loss(bbox_preds, bboxes)
        #     loss_bbox = (loss_bbox * mask).sum() / mask.sum()

        #     loss_cls = criterian(cls_logits, labels)
        #     loss = loss_cls + 0.01 * loss_bbox

        #     loss.backward()
        #     optimizer.step()

        # else:
        cls_logits = model(images)
        loss = criterian(cls_logits, labels)
        loss.backward()
        optimizer.step()  

        total_loss += loss.item() * labels.size(0)

        preds = (torch.sigmoid(cls_logits) > 0.5).float()
        # print(preds)
        total_correct += (preds == labels).sum().item()

        total_samples += labels.size(0)

        # print(f'Batch Loss: {loss.item():.4f}')

        if count % 500 ==0:
            print("Step", count)
        count+=1

        # break

    return total_loss / total_samples, total_correct / total_samples


def validate(model, val_loader, criterian, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for images, _, labels, _, _ in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss_cls = criterian(outputs, labels)

            loss = loss_cls

            total_loss += loss.item() * labels.size(0)

            preds = (torch.sigmoid(outputs)> 0.5).float()
            # print(preds)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples