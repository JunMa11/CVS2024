"""
multi-label classification with weighted binary cross entropy loss
python train_aug.py --img_size 512 -b 32 -m convnext_small.fb_in22k_ft_in1k --loss_type wbce
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
join = os.path.join
from tqdm import tqdm
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict
from cvs_datasets import CVSData
import argparse
import matplotlib.pyplot as plt
import random
from metrics import compute_overall_metrics

frames_path = 'data-CVS/train/frames'
val_frames_path = 'data-CVS/val/frames_val'
labels_path = 'data-CVS/train2/labels'

data_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
dataset = CVSData(frames_path=frames_path, labels_path=labels_path, transform=data_transform, auto_aug=True)
print('dataset size:', len(dataset))

img, label, video_name, frame_id, metadata = dataset[random.randint(0, len(dataset))]
print(f'Video {video_name}, frame {frame_id}')
print('Label of c1 c2 c3:', label)
print(f'Confidence aware labels: {metadata["confidence_aware_labels"]}')
image_width, image_height = img.shape[2], img.shape[1]
print(f'Original Image size: {image_width}x{image_height}')
plt.imshow(img.permute(1,2,0))
# plt.show()
plt.savefig('img_train_demo.png', bbox_inches='tight', dpi=300)
plt.close()

# Define a model for multi label classification.
class MultiLabelClassifierBaseline(nn.Module):
    def __init__(self, model_name='convnext_base.fb_in22k_ft_in1k', num_classes=3, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        # MLP as classification head
        if model_name.startswith('resnet'):
            self.backbone.fc = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(self.backbone.num_features, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes),
            )
        elif model_name.startswith('vit'):
            # Pre-trained ViT as backbone
            self.backbone.head = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(self.backbone.embed_dim, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes),
            )
        elif model_name.startswith('convnext'):
            self.backbone.head = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Conv2d(self.backbone.num_features, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, num_classes, kernel_size=1, stride=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
        elif model_name.startswith('tf_efficientnetv2'):
            self.backbone.classifier = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(self.backbone.num_features, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes),
            )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.backbone(x)  # Pass input image to ViT
        x = self.classifier(x)  # Pass through MLP classifier
        return x


class BCEwithClassWeights(nn.Module):
    def __init__(self, class_instance_nums, total_instance_num, device='cuda'):
        super(BCEwithClassWeights, self).__init__()
        class_instance_nums = torch.tensor(class_instance_nums, dtype=torch.float32)
        p = class_instance_nums / total_instance_num
        self.pos_weights = torch.exp(1-p).to(device)
        self.neg_weights = torch.exp(p).to(device)


    def forward(self, pred, label):
        # https://www.cse.sc.edu/~songwang/document/cvpr21d.pdf (equation 4)
        weight = label * self.pos_weights + (1 - label) * self.neg_weights
        loss = nn.functional.binary_cross_entropy_with_logits(pred, label, weight=weight)
        return loss


val_results = OrderedDict()
val_results['epochs'] = []
val_results['acc'] = []
val_results['f1'] = []
val_results['mAP'] = []
val_results['Brier-c1'] = []
val_results['Brier-c2'] = []
val_results['Brier-c3'] = []

def valid_one_epoch(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    overall_probabilities=[]
    overall_gt_labels=[]
    overall_confidence_aware_labels=[]
    overall_raw_gt_labels={
                        'c1':[],
                        'c2':[],
                        'c3':[]
                        }
    with torch.no_grad():
        for images, labels, video_name, frame_id, metadata in tqdm(dataloader):
            output = model(images.to(device))
            overall_probabilities.append(output.detach().cpu().numpy())
            # ground truth labels
            overall_gt_labels.append(labels.cpu().numpy())
            overall_confidence_aware_labels.append(np.transpose(np.array([metadata['confidence_aware_labels']['c1'],
                                                    metadata['confidence_aware_labels']['c2'],
                                                    metadata['confidence_aware_labels']['c3']])))
            for crit in ['c1','c2','c3']:
                overall_raw_gt_labels[crit].append(np.concatenate(np.expand_dims(metadata['raw_labels'][crit],0),1))


    for crit in ['c1','c2','c3']:
        overall_raw_gt_labels[crit] = np.transpose(np.concatenate(overall_raw_gt_labels[crit],1))
    overall_probabilities = np.concatenate(overall_probabilities)
    overall_gt_labels = np.concatenate(overall_gt_labels)
    overall_confidence_aware_labels = np.concatenate(overall_confidence_aware_labels)
    val_metrics = compute_overall_metrics(overall_gt_labels, overall_confidence_aware_labels, overall_probabilities)

    return val_metrics

# Example of use of the class
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ViT Multi-Label Classifier')
    # img size
    parser.add_argument('--img_size', type=int, default=512, help='image size')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--loss_type', type=str, default='wbce', help='loss type: bce, wbce, asl')
    parser.add_argument('--patience_epoch', type=int, default=10, help='stop training after patience')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('-m', '--model_name', type=str, default='convnext_base.fb_in22k_ft_in1k', help='time model names: vit_base_patch16_224')
    parser.add_argument('--save_model_path', type=str, default='/cluster/projects/bwanggroup/jma/data-CVS/checkpoints-alldata', help='path to save model')
    args = parser.parse_args()
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    tr_dataset = CVSData(frames_path=frames_path, labels_path=labels_path, transform=transform, auto_aug=True)
    tr_data_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    # Set device
    device = torch.device(args.device)

    # Initialize model
    model = MultiLabelClassifierBaseline(model_name=args.model_name).to(device)
    model_save_path = os.path.join(args.save_model_path, args.model_name + 'b' + str(args.batch_size)) + 'hw' + str(args.img_size) + args.loss_type + '-lr' + str(args.lr) 
    os.makedirs(model_save_path, exist_ok=True)
    # Define loss function
    if args.loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_type == 'wbce':
        criterion = BCEwithClassWeights(class_instance_nums=[1275, 2309, 1545], total_instance_num=len(tr_dataset), device=device)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Define number of epochs
    num_epochs = args.num_epochs

    # Create validation loader
    val_dataset = CVSData(frames_path=val_frames_path, labels_path=labels_path, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    losses = []
    best_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0.0

        for images, labels, _, _, _ in tqdm(tr_data_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(tr_data_loader):.4f}")
        losses.append(total_loss/len(tr_data_loader))

        # Plot losses
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(join(model_save_path, args.model_name + '_loss.png'))
        plt.close()

        val_metric = valid_one_epoch(model, val_dataloader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Metrics: {val_metric}")
        val_results['epochs'].append(epoch)
        val_results['acc'].append(val_metric['accuracy'])
        val_results['mAP'].append(val_metric['mAP'])
        val_results['f1'].append(val_metric['f1'])
        val_results['Brier-c1'].append(val_metric['brier_score']['c1'])
        val_results['Brier-c2'].append(val_metric['brier_score']['c2'])
        val_results['Brier-c3'].append(val_metric['brier_score']['c3'])
        val_results_df = pd.DataFrame(val_results)
        val_results_df.to_csv(join(model_save_path, args.model_name + '_val_results.csv'))
        
        # plot val acc, mAP, f1
        plt.plot(val_results['epochs'], val_results['acc'], label='Accuracy')
        plt.plot(val_results['epochs'], val_results['mAP'], label='mAP')
        plt.plot(val_results['epochs'], val_results['f1'], label='F1')
        # plt.plot(val_results['epochs'], val_results['Brier-c1'], label='Brier-c1')
        # plt.plot(val_results['epochs'], val_results['Brier-c2'], label='Brier-c2')
        # plt.plot(val_results['epochs'], val_results['Brier-c3'], label='Brier-c3')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Validation Metrics')
        plt.legend()
        plt.savefig(join(model_save_path, args.model_name + '_val_metrics.png'))
        plt.close()

        # Save model
        model_ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(tr_data_loader),
            }
        # torch.save(model_ckpt, os.path.join(model_save_path, 'latest_model.pth'))
        if total_loss / len(tr_data_loader) < best_loss:
            best_loss = total_loss / len(tr_data_loader)
            torch.save(model_ckpt, os.path.join(model_save_path, 'best_tr_loss_model.pth'))
            patience = 0
        else:
            # early stop. if loss doesn't decrease for three epochs, stop
            if (total_loss / len(tr_data_loader) - best_loss) < 0.001:
                patience += 1
                print(f'{epoch} patience: {patience} loss: {total_loss / len(tr_data_loader)}')
            if patience > args.patience_epoch:
                break
        # save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 and epoch > 10:
            torch.save(model_ckpt, os.path.join(model_save_path, f'epoch_{epoch}_model.pth'))
        
        # if best val mAP doesn't improve for patience epochs, stop
        if epoch> 10 and val_metric['mAP'] < max(val_results['mAP']):
            patience += 1
            if patience > args.patience_epoch:
                break
        else:
            patience = 0
            # new best mAP and save model
            torch.save(model_ckpt, os.path.join(model_save_path, 'best_val_model.pth'))
       
