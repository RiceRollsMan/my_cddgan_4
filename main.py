from train import train
from test import test


def main(dataset='xray'):
    modelConfig = None
    if dataset == 'xray':
        modelConfig = {
            "state": "train",  # train or test
            "epoch": 200,
            "batch_size": 32,
            "dataset": 'xray',
            # img
            "img_size": 256,
            "img_channel": 3,
            # label
            "label_channel": 1,
            "label_features_channel": 8,
            "label_seq_length": 128,
            "label_embedding_dim": 768,
            # latent
            "latent_channel": 4,
            "latent_dim": 64,
            # time_steps
            "num_time_steps": 4,
            # optimizer
            "lr_g": 1e-4,
            "lr_d": 5e-5,
            "beta_min": 0.1,
            "beta_max": 20,
            "use_geometric": False,
            # file
            'dataset_root': 'D:/Code/data/Xray/Xray',
            "save_weight_dir": "./Checkpoints_xray/",
            "save_img": "./save_images",
            "device": "cuda",
            "training_load_weight": None,
            "labels": "no acute cardiopulmonary disease . the heart , pulmonary and mediastinum are within normal "
                      "limits . there is no pleural effusion or pneumothorax . there is no focal air space opacity to "
                      "suggest a pneumonia .",
            "test_load_weight": "ckpt_199_.pt", }
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        test(modelConfig)


if __name__ == '__main__':
    main(dataset='xray')
