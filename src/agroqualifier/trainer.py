import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import tqdm

import wandb


class Trainer:
    def __init__(
        self,
        model,
        params,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device: str = "cuda",
        wandb_logging=False,
    ):
        self.model = model.to(device)
        self.params = params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        if wandb_logging:
            wandb.init(
                project="AgroQualifier",
                name=self.params.__class__.__name__,
                config={
                    "learning_rate": self.params.training_params.l_rate,
                    "architecture": self.params.model_params.architecture,
                    "dataset": (f"Use original_light, " if self.params.dataset_params.original_light else "")
                    + (f"Use IR_lamp_light" if self.params.dataset_params.IR_lamp_light else ""),
                    "epochs": self.params.training_params.num_epochs,
                },
            )

    def train(self):
        for epoch in range(self.params.training_params.num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in tqdm.tqdm(self.train_loader, desc=f"Train loop, Epoch {epoch + 1}"):
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # Print average training loss for the epoch
            training_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{self.params.training_params.num_epochs}, Training Loss: {training_loss}")
            wandb.log({"Train/loss": training_loss, "Learning rate": self.get_lr()})

            if self.scheduler is not None:
                self.scheduler.step()

            # Evaluate the model on the validation set
            self.evaluate()

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(self.val_loader, desc="Validation loop"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                val_loss += self.criterion(outputs, labels).item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        val_loss = val_loss / len(self.val_loader)
        wandb.log({"Validation/loss": val_loss, "Validation/accuracy": accuracy})
        print(f"Validation Accuracy: {accuracy}, Validation loss: {val_loss}")

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']