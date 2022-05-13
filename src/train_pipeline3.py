import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler,autocast

import os
import json
import random
from tqdm import tqdm


from dataset import BirdClif22DatasetV2
from models.factory import Activation
from utils import get_scheduler,get_optimizer,get_loss,MicroF1,MacroF1,AverageMeter
from transforms import get_tfm,Mixup,addFreqCoords
from models import create_sed_model



def initialize_params(cfg):
    device = cfg['training']['device']


    train_dataset = BirdClif22DatasetV2(
        split='train',
        **cfg['dataset']
    )
    val_dataset = BirdClif22DatasetV2(
        split='val',
        **cfg['dataset']
    )

    keys = ['nfft','hop_length','n_mels','sample_rate','stype','normalize']
    val_tfm_kwargs = {k:cfg['transform']['kwargs'][k] for k in keys}

    transform = get_tfm(level=cfg['transform']['level'],**cfg['transform']['kwargs'])

    val_transform = get_tfm(level=0,**val_tfm_kwargs)
    transform.to(device)
    val_transform.to(device)



    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=cfg['training']['batch_size'],
                                  shuffle=True,
                                  sampler=None,
                                  batch_sampler=None,
                                  num_workers=2)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=1,
                                shuffle=True,
                                sampler=None,
                                batch_sampler=None,
                                num_workers=2)

    scaler = GradScaler() if cfg['training']['amp'] else None

    global max_grad_norm, accum_steps, num_splits, min_crop, cow_criterion, use_mixup,use_coordconv, scored_indices
    max_grad_norm= cfg['training']['max_grad_norm']
    accum_steps = cfg['training']['accumulation_steps']
    num_splits = cfg['dataset']['crop_length'] // 5
    min_crop = cfg['transform']['kwargs']['sample_rate'] * 5
    use_coordconv = cfg['training']['use_coord_conv']
    scored_indices = torch.tensor([3, 6, 7, 9, 44, 46, 47, 60, 62, 63, 64, 65, 67, 70, 72, 90, 101, 111, 131, 141, 150],requires_grad=False).to(device)


    use_mixup = cfg['training']['use_mixup']
    if use_mixup:
        global mixup
        mixup = Mixup(0.5) 
        mixup.to(device)


    cow_criterion = get_loss(name='cow_l1_loss')


    model = create_sed_model(**cfg['model'])

    optimizer = get_optimizer(
        params=model.parameters(),
        name=cfg['optimizer']['name'],
        **cfg['optimizer']['kwargs'])

    scheduler = get_scheduler(
        optimizer=optimizer,
        name=cfg['scheduler']['name'],
        **cfg['scheduler']['kwargs'])

    
    #val_transform.eval()

    criterion = get_loss(name=cfg['loss']['name'],**cfg['loss']['kwargs'])
    activation = Activation('sigmoid')

    model.to(device)
    criterion.to(device)
    cow_criterion.to(device)
    activation.to(device)
    
    return model,criterion,optimizer,scheduler,activation,train_dataloader,val_dataloader,scaler,transform,val_transform


def update_meters(d,scores,instant=False):
    for k,v in scores.items():
        if k not in d.keys():
            d[k] = AverageMeter() 
        d[k].update(v)
    return d

def decompress(items,device='cuda'):
    x = items['x'].to(device)
    y_gt = items['y_gt'].to(device)
    weights = items['weights'].to(device)
    rating = items['rating_weight'].to(device)
    cow_weights = items['cow_weights'].to(device)



    return x,y_gt,weights,rating,cow_weights




def train_epoch(model,criterion,optimizer,train_dataloader,activation,transform,epoch = 0,scaler=None,device='cuda',amp=True):
    model.train()
    model.zero_grad()
    #print(transform)
    loader = tqdm(train_dataloader)
    criterion_name = criterion.__name__
    criterion_name2 = cow_criterion.__name__

    loss_meter = AverageMeter()
    loss_meter2 = AverageMeter()

    microf1_metric,macrof1_metric = MicroF1(0.5),MacroF1(0.5) 

    for batch_idx,items in enumerate(loader):
        x,y_gt,weights,rating,cow_weights = decompress(items,device=device)
        #print(weights)
        with torch.no_grad():
            x = transform(x)
            if use_coordconv:
                x = addFreqCoords(x)

        if use_mixup and random.random() < 0.5:
            x,y_gt,rating,cow_weights,weights = mixup(x,y_gt,rating,cow_weights,weights)

        with autocast(enabled=amp):
            y_out,cow_weights_p = model(x)
            y_pred = activation(y_out)


            if criterion_name in ['focal_loss','bce_loss']:
                loss = criterion(y_out,y_gt)

            else:
                loss = criterion(y_out,y_gt,weights,rating)
 

            cow_loss = cow_criterion(cow_weights_p,cow_weights)

        loss_value = loss.item()
  
        closs_value = cow_loss.item()

        loss = loss  + 0.05 * cow_loss 
        loss = loss / accum_steps

        if amp:
            scaler.scale(loss).backward()
       
        else:
            loss.backward()
        
        
            
        if ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(loader)):
            if amp:
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) 
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) 
                optimizer.step()
            
            optimizer.zero_grad()

        y_pred = y_pred.detach()
        
        loss_meter.update(loss_value)

        loss_meter2.update(closs_value)
        loss_value = loss_meter.get_update()
        closs_value = loss_meter2.get_update()


        microf1_score, class_microf1_scores= microf1_metric(y_pred,y_gt)
        scored_f1 = torch.gather(class_microf1_scores,dim=0,index=scored_indices).mean().item() * 100
        microf1_score = microf1_score.item() * 100
        macrof1_score = macrof1_metric(y_pred,y_gt).item() * 100

        s = f'TrainEpoch[{epoch}]: {criterion_name} : {loss_value:.5},{criterion_name2} : {closs_value:.5} ,MicroF1: {microf1_score:.5}% , MacroF1 : {macrof1_score:.5}%, ScoredF1 : {scored_f1:.5}%'
        #for k,v in loss_meters.items():
        #    s+= f' {k} : {v.get_update():.5} -'
        #s += f' bbox_iou : {iou_meter.get_update()*100:.5} -'

        loader.set_postfix_str(s)
        #break

    s+='\n'
    return s

def validate_epoch(model,criterion,val_dataloader,activation,transform,epoch = 0,device='cuda'):
    model.eval()
    loader = tqdm(val_dataloader)
    criterion_name = criterion.__name__
    criterion_name2 = cow_criterion.__name__

    loss_meter = AverageMeter()
    loss_meter2 = AverageMeter()

    
    microf1_metric,macrof1_metric = MicroF1(0.5),MacroF1(0.5) 

    for batch_idx,items in enumerate(loader):
        #break
        x,y_gt,weights,rating,cow_weights = decompress(items,device=device)
        cow_weights = cow_weights[0,...]
        x = x.permute(1,0,2)

        with torch.no_grad():
            x = transform(x)
            if use_coordconv:
                x = addFreqCoords(x)

            y_out,cow_weights_p = model(x)
            y_out,_ = torch.max(y_out,dim=0)
            y_out = y_out.unsqueeze(0)
            y_pred = activation(y_out)

            if criterion_name in ['focal_loss','bce_loss']:
                loss = criterion(y_out,y_gt)

            else:
                loss = criterion(y_out,y_gt,weights,rating)


            cow_loss = cow_criterion(cow_weights_p,cow_weights)

        loss_value = loss.item()

        closs_value = cow_loss.item()

        loss = loss  + 0.05 * cow_loss

        y_pred = y_pred.detach()
        
        
        loss_meter.update(loss_value)

        loss_meter2.update(closs_value)
        loss_value = loss_meter.get_update()
        closs_value = loss_meter2.get_update()


        microf1_score, class_microf1_scores= microf1_metric(y_pred,y_gt)
        scored_f1 = torch.gather(class_microf1_scores,dim=0,index=scored_indices).mean().item() * 100
        microf1_score = microf1_score.item() * 100
        macrof1_score = macrof1_metric(y_pred,y_gt).item() * 100

        s = f'ValEpoch[{epoch}]: {criterion_name} : {loss_value:.5}, {criterion_name2} : {closs_value:.5} ,MicroF1 : {microf1_score:.5}% , MacroF1 : {macrof1_score:.5}%, ScoredF1 : {scored_f1:.5}%'
        loader.set_postfix_str(s)
        #break
    scores = {'MicroF1' : microf1_score,'MacroF1' : macrof1_score, 'ScoredF1' : scored_f1}
    s+='\n'
    return s,scores


def save_model(path,model,epoch,score,lr):
    if not path.endswith('.pth'):
        path += '/best_model.pth'
    torch.save(
        {
            'state_dict' : model.state_dict(),
            'epoch' : epoch,
            'best_score' : score,
            'learning_rate' : lr
        },
        path)


def save_configs(path, configs):
    f = open(os.path.join(path, 'configs.json'), "w+")
    json.dump(configs, f, indent=3)
    f.close()

def save_logs(path,logs):
    with open(path+'/logs.txt','w+') as f:
        f.write(logs)

def make_dir(path):
    try:
        os.makedirs(path)
    except:
        pass

def train_model(cfg):

    ###some configs###
    save_dir = cfg['save_dir']
    num_epochs = cfg['training']['epochs']
    device = cfg['training']['device']
    amp = cfg['training']['amp']
    val_freq = cfg['training']['val_freq']
    make_dir(save_dir)
    save_configs(save_dir,cfg)
    ###___________###
    model,criterion,optimizer,scheduler,activation,train_dataloader,val_dataloader,scaler,transform,val_transform= initialize_params(cfg)

    best_score = -9999.99
    best_micro_score = -99999.99
    best_scored_f1 = -9999.99
    logs = ''
    for epoch in range(num_epochs):
        lr = optimizer.param_groups[0]['lr']
        s_train = train_epoch(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        activation=activation,
                        transform=transform,
                        train_dataloader=train_dataloader,
                        scaler=scaler,
                        amp=amp,
                        device=device
                        )
        logs+= '>>>' + s_train
        if (epoch+1) % val_freq  == 0 or (epoch+1) == num_epochs:
            s_val,scores = validate_epoch(
                                        epoch= epoch,
                                        model=model,
                                        criterion=criterion,
                                        activation=activation,
                                        transform=val_transform,
                                        val_dataloader=val_dataloader,
                                        device=device
                                        )
            logs+= '>>>' + s_val
            score = scores['MacroF1']

            if score >= best_score:
                s_save = f'Current MacroF1 [{score:.5}%] is better than previous best MacroF1 [{best_score:.5}%] ---> Saving Model!!!'
                best_score=score
                print(s_save)
                logs+= '>>>' + s_save + '\n'
                save_model(save_dir,model,epoch,score,lr)
            
            mscore = scores['MicroF1']

            if mscore >= best_micro_score:
                s_save = f'Current MicroF1 [{mscore:.5}%] is better than previous best MicroF1 [{best_micro_score:.5}%] ---> Saving Model!!!'
                best_micro_score=mscore
                print(s_save)
                logs+= '>>>' + s_save + '\n'
                save_model(save_dir+'/best_micro_model.pth',model,epoch,mscore,lr)
            
            sscore = scores['ScoredF1']

            if sscore >= best_scored_f1:
                s_save = f'Current ScoredF1 [{sscore:.5}%] is better than previous best ScoredF1 [{best_scored_f1:.5}%] ---> Saving Model!!!'
                best_scored_f1=sscore
                print(s_save)
                logs+= '>>>' + s_save + '\n'
                save_model(save_dir+'/best_scored_model.pth',model,epoch,sscore,lr)


                


        save_model(save_dir+'/last_model.pth',model,epoch,score,lr)


        save_logs(save_dir,logs)

        scheduler.step()


