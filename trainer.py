import os
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from data.utils import get_dataloader
from model.model import BBoxDetector
from eval.eval_iou3d import eval_iou3d

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

class Trainer:

    def __init__(self, learning_rate=0.001, epochs=1, model_path="outputs"):
        self.epochs = epochs

        self.train_dataloader = get_dataloader(split="train")
        self.test_dataloader = get_dataloader(split="test")

        self.model = BBoxDetector()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        os.path.join(os.getcwd(), model_path)
        os.makedirs(model_path, exist_ok=True)

        self.tb_writer = None
        if TENSORBOARD_FOUND:
            self.tb_writer = SummaryWriter(model_path)
        else:
            print("Tensorboard not available: not logging progress")

    def train(self):
        print("Start training")
        self.model.train()

        iteration = 0
        for epoch in range(self.epochs):
            for i, batch in enumerate(self.train_dataloader):
                
                self.optimizer.zero_grad()
                
                image = batch[0].to(self.device).float()
                bbox_gt = batch[1]['bbox'].to(self.device).float()
                bbox_2d = batch[2].to(self.device).float()
                # pc = batch[3].to(self.device)

                bbox_pred = self.model(image, bbox_2d)

                bbox_loss = F.l1_loss(bbox_pred, bbox_gt)

                loss = bbox_loss
                
                loss.backward()
                self.optimizer.step()
                if i % 10 == 0 and self.tb_writer:
                    self.tb_writer.add_scalar('Loss', loss.item(), iteration)

                iteration += 1
                
        print("Training Done")
        
    def test(self):
        print("Start testing")
        self.model.eval()
        
        for i, batch in enumerate(self.test_dataloader):
            image = batch[0].to(self.device).float()
            bbox_gt = batch[1]['bbox'].to(self.device).float()
            bbox_2d = batch[2].to(self.device).float()
            
            bbox_pred = self.model(image, bbox_2d)

            bbox_pred = bbox_pred.squeeze(0)
            bbox_gt = bbox_gt.squeeze(0)

            ious = eval_iou3d(bbox_pred, bbox_gt)
            print(f"iou for {i} ", ious)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--lr", type=float, required=False, default=0.001)
    parser.add_argument("--epochs", type=int, required=False, default=1)
    parser.add_argument("--model_path", type=str, required=False, default="outputs")
    args = parser.parse_args()

    trainer = Trainer(args.lr, args.epochs, args.model_path)
    trainer.train()
    trainer.test()