from pathlib import Path
import torch
import torch.nn as nn
import os 

@torch.no_grad()

def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
current_dir = os.path.dirname(os.path.abspath(__file__))
clean_model_savedir = os.path.join(current_dir,"results", 'models', 'clean')
dirty_model_savedir = os.path.join(current_dir,"results", 'models', 'dirty')

Path(clean_model_savedir).mkdir(parents=True, exist_ok=True)
Path(dirty_model_savedir).mkdir(parents=True, exist_ok=True)




def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,weight_decay=0, 
        grad_clip=None, opt_func=torch.optim.SGD,train_type='clean'):
    history = []
    optimizer = opt_func
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    for epoch in (range(epochs)):
        # Training Phase 
        model.train()
        train_losses = []
        train_accuracy= []
        lrs=[]
        for (batch_idx, batch) in enumerate(train_loader):
            loss,accuracy = model.training_step(batch)
            train_losses.append(loss)
            train_accuracy.append(accuracy)
            loss.backward()
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
            if batch_idx % 60 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.4f}'.
                format(epoch+1, batch_idx , len(train_loader),
                       100. * batch_idx / len(train_loader), loss,accuracy))
            
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_accuracy'] = torch.stack(train_accuracy).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)



        #Use pytorch to save this model chekpoint in model_save_dir
        if train_type == 'clean':
            torch.save(model.state_dict(), f"{clean_model_savedir}/clean_ckpt_{epoch}.pth")
        else:
            torch.save(model.state_dict(), f"{dirty_model_savedir}/dirty_ckpt_{epoch}.pth")

        
    return history
