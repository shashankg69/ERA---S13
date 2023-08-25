import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from YOLOv3_model import YOLOv3
from dataset import YOLODataset
from loss import YoloLoss
from torch import optim
from torch.utils.data import DataLoader

import config
from utils import ResizeDataLoader, cells_to_bboxes, non_max_suppression


class Model(LightningModule):
    def __init__(self, in_channels=3, num_classes=config.NUM_CLASSES, batch_size=config.BATCH_SIZE,
                 learning_rate=config.LEARNING_RATE, num_epochs=config.NUM_EPOCHS):
        super(Model, self).__init__()
        self.network = YOLOv3(in_channels, num_classes)
        self.criterion = YoloLoss()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.register_buffer("scaled_anchors", config.SCALED_ANCHORS)

    def forward(self, x):
        return self.network(x)

    def common_step(self, batch):
        x, y = batch
        out = self.forward(x)
        loss = self.criterion(out, y, self.scaled_anchors)
        del out, x, y
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log(f"train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log(f"val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, (tuple, list)):
            x, _ = batch
        else:
            x = batch
        return self.forward(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate/100, weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            steps_per_epoch=len(self.train_dataloader()),
            epochs=self.num_epochs,
            pct_start=0.2,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy='linear'
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
    
     # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     out = self.forward(x)
    #     loss = self.criterion(out, y, self.scaled_anchors)
    #     self.log(f"Train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     out = self.forward(x)
    #     loss = self.criterion(out, y, self.scaled_anchors)
    #     self.log(f"Validation_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    #     return loss

    def on_epoch_end(self):
        self.model.eval()
        loader = self.train_dataloader()
        thresh = 0.6
        iou_thresh = 0.5

        x, y = next(iter(loader))
        x = x.to(self.device)

        with torch.no_grad():
            out = self.model(x)
            bboxes = [[] for _ in range(x.shape[0])]
            for i in range(3):
                batch_size, A, S, _, _ = out[i].shape
                anchor = self.anchors[i]
                boxes_scale_i = cells_to_bboxes(
                    out[i], anchor, S=S, is_preds=True
                )
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box

        self.model.train()

        for i in range(batch_size // 4):
            nms_boxes = non_max_suppression(
                bboxes[i],
                iou_threshold=iou_thresh,
                threshold=thresh,
                box_format="midpoint",
            )
            self.plot_image(x[i].permute(1, 2, 0).detach().cpu(), nms_boxes)

            

    def train_dataloader(self):
        train_dataset = YOLODataset(
            config.DATASET + '/train.csv',
            transform=config.train_transforms,
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
            mosaic=0.75
        )

        train_loader = ResizeDataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=True,
            resolutions=config.MULTIRES,
            cum_weights=config.CUM_PROBS
        )

        return train_loader

    def val_dataloader(self):
        train_eval_dataset = YOLODataset(
            config.DATASET + '/test.csv',
            transform=config.test_transforms,
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
            mosaic=0
        )

        train_eval_loader = DataLoader(
            dataset=train_eval_dataset,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=False
        )

        return train_eval_loader


def main():
    num_classes = 20
    IMAGE_SIZE = 416
    INPUT_SIZE = IMAGE_SIZE * 2
    model = Model(num_classes=num_classes)
    from torchinfo import summary
    print(summary(model, input_size=(2, 3, INPUT_SIZE, INPUT_SIZE)))
    inp = torch.randn((2, 3, INPUT_SIZE, INPUT_SIZE))
    out = model(inp)
    assert out[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert out[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert out[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")


if __name__ == "__main__":
    main()
    



   