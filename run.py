import os
import cv2
import torch.distributed as dist
from torchvision import transforms
from torchvision import models
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import shutil


class ImageDataSet(Dataset):
    def __init__(self, train, test, val):
        attributes = pd.read_csv(r"./archive/list_attr_celeba.csv")
        attributes = attributes.replace(-1, 0)
        partition_df = pd.read_csv(r"./archive/list_eval_partition.csv")
        self.dataset = attributes.join(partition_df.set_index('image_id'), on='image_id')
        if train:
            self.dataset = self.dataset.loc[self.dataset['partition'] == 0]
            self.images = self.dataset['image_id']
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((299, 299)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
            ])
        elif test:
            self.dataset = self.dataset.loc[self.dataset['partition'] == 1]
            self.images = self.dataset['image_id']
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
        elif val:
            self.dataset = self.dataset.loc[self.dataset['partition'] == 2]
            self.images = self.dataset['image_id']
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
            ])
        self.len = len(self.dataset)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = cv2.imread(r"./archive/img_align_celeba/img_align_celeba/" + self.images.iloc[index])
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        atrributes = torch.from_numpy(np.array(self.dataset.iloc[index, 1:41], dtype=np.int32))
        image_id = self.dataset.iloc[index, 0:1].tolist()

        return {
            'image': image,
            'attributes': atrributes,
            'image_id': image_id
        }


attributes_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                   'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                   'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                   'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                   'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                   'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                   'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                   'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                   'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                   'Wearing_Necktie', 'Young']

dataset_train = ImageDataSet(train=True, test=False, val=False)
dataset_test = ImageDataSet(train=False, test=True, val=False)
dataset_val = ImageDataSet(train=False, test=False, val=True)

dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test, dataset_val])


def predict(rank, size):
    files = os.listdir('query')
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])
    image = cv2.imread('query/' + files[0])
    image = transform(image)
    #
    resnet18 = models.resnet18(pretrained=False)
    num_final_in = resnet18.fc.in_features
    NUM_FEATURES = 40
    resnet18.fc = nn.Sequential(nn.Linear(num_final_in, NUM_FEATURES), nn.Sigmoid())

    model = DDP(resnet18)

    checkpoint = torch.load(r"./model.checkpoint", map_location='cpu')
    # model.load_state_dict(torch.load(r"./model.checkpoint"))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    prediction = model(image[None, ...])

    output_dict = {}

    sampler = DistributedSampler(dataset, num_replicas=size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler)

    for train_i, train_data in enumerate(dataloader):
        tr_data, tr_target, tr_image_id = train_data['image'], train_data['attributes'], train_data[
            'image_id']

        cos = torch.nn.CosineSimilarity(dim=1)
        output = cos(prediction, tr_target)
        if output >= 0.85:
            print("Cosine Similarity with ", ",", tr_image_id[0], " :", output)
            output_dict[tr_image_id[0]] = [output.item(), tr_target]

    a = sorted(output_dict.items(), key=lambda x: x[1][0], reverse=True)
    a = a[:3]
    actual_attributes = pd.read_csv(r"./archive/list_attr_celeba.csv")
    actual_attributes = actual_attributes.replace(-1, 0)
    actual_attributes = list(actual_attributes.loc[actual_attributes['image_id'] == files[0]].iloc[0])
    print("Actual attributes: ")
    for i in range(len(attributes_list)):
        print(attributes_list[i], ": ", actual_attributes[i+1])
    print("Predicted attributes: ")
    for i in range(len(attributes_list)):
        print(attributes_list[i], ": ", prediction[0][i].item())
    for each_ in a:
        image = each_[0]
        image_file = cv2.imread('./archive/img_align_celeba/img_align_celeba/' + image[0])
        attributes = np.array(attributes_list)[np.array(each_[1][1].numpy()[0] == 1)]
        print("Attributes for image ", image[0], ": ", attributes)
        cv2.imwrite('./results/' + image[0], image_file)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'mars'
    os.environ['MASTER_PORT'] = '29700'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    # torch.manual_seed(40)


if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    setup(int(sys.argv[1]), int(sys.argv[2]))
    # train(int(sys.argv[1]), int(sys.argv[2]))
    if not os.path.isdir('query'):
        os.mkdir('query')
    files = os.listdir('query')
    if len(files) != 1:
        print('Need 1 image in query folder')
    if not os.path.isdir('results'):
        os.mkdir('results')
    files_results = os.listdir('./results')
    if len(files_results) != 0:
        print("Result directory should be empty")
        sys.exit(1)
    predict(int(sys.argv[1]), int(sys.argv[2]))
