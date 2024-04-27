"""Module containing dataloader related functionality."""

import torch
from torch.utils.data import DataLoader

from transoar.data.dataset import TransoarDataset
from transoar.utils.bboxes import segmentation2bbox


def get_loader(config, split, batch_size=None, test_script=False):

    if not batch_size:
        batch_size = config['batch_size']

    # Init collator
    collator = TransoarCollator(config, split)
    shuffle = False if split in ['test', 'val'] else config['shuffle']

    if test_script:
        dataset = TransoarDataset(config, split, dataset=1, selected_samples=None, test_script=True)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=config['num_workers'], collate_fn=collator
        )

    elif config["CL_reg"] is False and config["CL_replay"] is False: # Normal training
        dataset = TransoarDataset(config, split)

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=config['num_workers'], collate_fn=collator
        )
    elif config["CL_reg"] is True and config["CL_replay"] is False: # Training with CL_reg method
        if split == 'test': # Test on both datasets to see the evolution of performance with CL_reg method
            dataset_1 = TransoarDataset(config, split)
            dataloader_1 = DataLoader(
                dataset_1, batch_size=batch_size, shuffle=shuffle,
                num_workers=config['num_workers'], collate_fn=collator
            )
            dataset_2 = TransoarDataset(config, split, dataset=2)
            dataloader_2 = DataLoader(
                dataset_2, batch_size=batch_size, shuffle=shuffle,
                num_workers=config['num_workers'], collate_fn=collator
            )
            dataloader = (dataloader_1, dataloader_2)

        else: # Train and validation on the first dataset
            dataset = TransoarDataset(config, split)

            if config["mixing_datasets"]:
                shuffle = False

            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle,
                num_workers=config['num_workers'], collate_fn=collator
            )
    elif config["CL_reg"] is False and config["CL_replay"] is True: # Training with CL_replay method
        if split == 'test':
            dataset_1 = TransoarDataset(config, split)
            dataloader_1 = DataLoader(
                dataset_1, batch_size=batch_size, shuffle=shuffle,
                num_workers=config['num_workers'], collate_fn=collator
            )

            dataset_2 = TransoarDataset(config, split, dataset=2)
            dataloader_2 = DataLoader(
                dataset_2, batch_size=batch_size, shuffle=shuffle,
                num_workers=config['num_workers'], collate_fn=collator
            )
            dataloader = (dataloader_1, dataloader_2)

        elif split == 'train':
            collator = TransoarCollator(config, split, CL_replay=True)
            dataset = TransoarDataset(config, split, dataset=2)
            dataloader = DataLoader(
                dataset, batch_size=1, shuffle=shuffle,
                num_workers=config['num_workers'], collate_fn=collator
            )


        elif split == 'val':
            dataset = TransoarDataset(config, split)

            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle,
                num_workers=config['num_workers'], collate_fn=collator
            )
    else:
        raise ValueError("Invalid config: CL_reg and CL_replay cannot be True at the same time.")

    return dataloader

def get_loader_CLreplay_selected_samples(config, split, batch_size=None, selected_samples=None):
    if not batch_size:
        batch_size = config['batch_size']

    # Init collator
    collator = TransoarCollator(config, split)
    shuffle = False 

    dataset = TransoarDataset(config, split, dataset=1, selected_samples=selected_samples)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=config['num_workers'], collate_fn=collator
    )

    return dataloader

# def init_fn(worker_id):
#     """
#     https://github.com/pytorch/pytorch/issues/7068
#     https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
#     """
#     torch_seed = torch.initial_seed()
#     if torch_seed >= 2**30:
#         torch_seed = torch_seed % 2**30
#     seed = torch_seed + worker_id

#     random.seed(seed)   
#     np.random.seed(seed)
#     monai.utils.set_determinism(seed=seed)
#     torch.manual_seed(seed)


class TransoarCollator:
    def __init__(self, config, split, CL_replay=False):
        self._bbox_padding = config['bbox_padding']
        self._split = split
        self.CL_replay = CL_replay

    def __call__(self, batch):
        batch_images = []
        batch_labels = []
        batch_masks = []
        if self._split == 'test' or self.CL_replay:
            batch_paths = []
            for image, label, path in batch:
                batch_images.append(image)
                batch_labels.append(label)
                batch_masks.append(torch.zeros_like(image))
                batch_paths.append(path)
        else:
            for image, label in batch:
                batch_images.append(image)
                batch_labels.append(label)
                batch_masks.append(torch.zeros_like(image))

        # Generate bboxes and corresponding class labels
        batch_bboxes, batch_classes = segmentation2bbox(torch.stack(batch_labels), self._bbox_padding)
        # print("batch_bboxes, batch_classes", batch_bboxes, batch_classes)
        # quit()

        if self._split == 'test' or self.CL_replay:
            return torch.stack(batch_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)), torch.stack(batch_labels), batch_paths    
        return torch.stack(batch_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)), torch.stack(batch_labels)
