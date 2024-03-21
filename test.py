import os

import torch
from torchvision import transforms

from model.generator import T2IGenerator
from transformers import BertTokenizer, BertModel


def getLabelsEncoding(labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    encoded_input = tokenizer(labels, return_tensors='pt')
    labelsEncoding = model(**encoded_input)
    expand_label = torch.zeros([1, 512, labelsEncoding.last_hidden_state.data.shape[2]])
    expand_label[:, :labelsEncoding.last_hidden_state.data.data.shape[1], :] = labelsEncoding.last_hidden_state.data

    return expand_label


def test(modelConfig):
    with torch.no_grad():
        device = torch.device(modelConfig['device'])
        generator = T2IGenerator(modelConfig=modelConfig).to(device)
        ckpt_G = torch.load(os.path.join(
            modelConfig['save_weight_dir'], 'generator/', modelConfig['test_load_weight']), map_location=device)
        generator.load_state_dict(ckpt_G)
        print('generator load weight done.')
        print(modelConfig['labels'])
        labels = getLabelsEncoding(modelConfig['labels']).to(device).reshape(1, 1, 512, 768)
        print(labels)
        x_0_predict = generator(labels).to(device)
        print(x_0_predict)
        img_pil = transforms.ToPILImage()(x_0_predict.reshape([3, 256, 256]))
        img_pil.show()
