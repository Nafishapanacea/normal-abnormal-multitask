import torch

# # if some images has no bbox, then there is a need  custom collate_fn
# def collate_fn(batch):
#     images, labels, bboxes = zip(*batch)

#     images = torch.stack(images)
#     labels = torch.stack(labels)

#     # keep bbox list (some None, some tensor)
#     return images, labels, bboxes


def train_one_epoch(model, train_loader, optimizer, criterian, bbox_loss, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for images, labels, bboxes, has_bbox in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        bboxes = bboxes.float().to(device)
        
        optimizer.zero_grad()

        outputs = model(images)                 # classification always
        loss_cls = criterian(outputs, labels)

        if has_bbox.any():
            outputs = model.backbone(images)
            bbox_preds = model.bbox_head(outputs[has_bbox])
            bbox_targets = bboxes[has_bbox]
            loss_bbox = bbox_loss(bbox_preds, bboxes)
        else:
            loss_bbox = torch.tensor(0.0, device=device)
        
        loss = loss_cls + loss_bbox

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = (torch.sigmoid(outputs) > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def validate(model, val_loader, criterian, bbox_loss, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for images, labels, bboxes, has_bbox in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            bboxes = bboxes.float().to(device)

            outputs = model(images)
            loss_cls = criterian(outputs, labels)
            
            if has_bbox.any():
                outputs = model.backbone(images)
                bbox_preds = model.bbox_head(outputs[has_bbox])
                bbox_targets = bboxes[has_bbox]
                loss_bbox = bbox_loss(bbox_preds, bboxes)
            else:
                loss_bbox = torch.tensor(0.0, device=device)
            
            loss = loss_cls + loss_bbox

            total_loss += loss.item() * labels.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples