from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch

DEFAULT_PAD_TOKEN = "<|pad|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_UNK_TOKEN = "<|unk|>"

def get_collated_dataset(tokenizer, dataframe, stepend_token: str):
    tokenizer.padding_side = "right"
    inputs = tokenizer(dataframe['input'], padding='longest', return_tensors="pt")
    label_ids = tokenizer(dataframe['label'], padding='longest', return_tensors="pt")
    if inputs['input_ids'].size() != label_ids['input_ids'].size():
        return ValueError(
            "The tensor.size() of input_ids for inputs and labels are not equal"
        )
    
    input_ids_list = inputs['input_ids'].tolist()
    label_ids_list = label_ids['input_ids'].tolist()
    """
    input_ids -->
    torch.tensor([
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0]]) 

    input_ids_list -->
    [
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0]
    ]
    """
    bar = tqdm(zip(input_ids_list, label_ids_list), total=len(input_ids_list), desc="Generating Labels -> ")
    labels_list = []
    for input, label in bar:
        temp_labels = []
        for input_idx, label_idx in zip(input, label):
            if input_idx == tokenizer.encode(stepend_token)[1]:
                temp_labels.append(label_idx)
            else:
                temp_labels.append(-100)
                
        labels_list.append(temp_labels)
    
    labels = torch.tensor(labels_list)
    
    if labels.size() != inputs['input_ids'].size():
        return ValueError(
            "The tensor.size() of inputs and labels are not equal"
        )
    
    return dict(input_ids=inputs['input_ids'], labels=labels, attention_mask=inputs['attention_mask'])

def get_dataloader(collated_dataset, world_size, local_rank, shuffle, seed, batch_size, drop_last):
    cated_dataset = TensorDataset(
        collated_dataset['input_ids'], collated_dataset['labels'], collated_dataset['attention_mask']
    )
    
    """
    cated_data ->
    (tensor("input_ids"), tensor("labels"), tensor("attention_mask"))
    """
    
    sampler = DistributedSampler(
        cated_dataset, num_replicas=world_size, rank=local_rank, shuffle=shuffle, seed=seed
    )

    loader = DataLoader(
        cated_dataset, shuffle=False, pin_memory=True, drop_last=drop_last, batch_size=batch_size,
        collate_fn=None, sampler=sampler
    )
    
    """
    output structure for my understanding, 
    e.g. batch_size = 2, dataset_size = 4 ->
    [
        [tensor(["input_ids1","input_ids2]), 
         tensor(["labels1","labels2"]), 
         tensor(["attention_mask1","attention_mask2"])],
        [tensor(["input_ids1","input_ids2]), 
         tensor(["labels1","labels2"]), 
         tensor(["attention_mask1","attention_mask2"])]
    ]
         
    """

    return sampler, loader
