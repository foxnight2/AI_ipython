import torch 
import torch.nn as nn
from functools import partial

from model import Encoder, Decoder, Pix2Seq
from dataset import Tokenizer, tokenizer_collate_fn, get_coco_dataset, train_transforms



if __name__ == '__main__':

    tokenizer = Tokenizer(91, 2000, spatial_size=(640, 640))

    enc = Encoder()
    dec = Decoder(tokenizer.vocab_size, tokenizer.max_len, 256, 8, 6)

    device = torch.device('cuda:1')
    m = Pix2Seq(enc, dec).to(device)


    root = '/paddle/dataset/det/train2017/'
    anno_file = '/paddle/dataset/det/annotations/instances_minitrain2017.json'

    dataset = get_coco_dataset(root, anno_file, transforms=train_transforms)

    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=2, 
                                            shuffle=True, 
                                            num_workers=2,
                                            collate_fn=partial(tokenizer_collate_fn, tokenizer=tokenizer))


    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_TOKEN)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)



    for e in range(12):
        for i, (images, tokens) in enumerate(dataloader):
            
            images = images.to(device) 
            tokens = tokens.to(device)
            
            y_input = tokens[:, :-1]
            y_expect = tokens[:, 1:]
            
            preds: torch.Tensor = m(images, y_input)

            loss = criterion(preds.permute(0, 2, 1),  y_expect)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(e, i, loss)
    

        if e % 2 == 0:
            torch.save(m.state_dict(), f'{e}.pth')
