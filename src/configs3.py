from dataset.bird_clif22_dataset import class_names,num_classes




epochs = 30
learning_rate = 0.0001
fold = 1

loss_name = 'focalv3'#'keypoint_mse_loss'
backbone = 'resnest50d'#'tf_efficientnetv2_s_in21ft1k''convnext_tiny_in22ft1k'
stride = 1

crop_length = 30
sampling_rate = 32000
n_fft = 2048
hop_length = 714
n_mels=224

freq_coord_conv = False
model = 'sed_v7'


cfg = {
    'model' : {
        'name' : model,
        'backbone' : backbone,
        'in_channels' : 1 + int(freq_coord_conv),
        'pretrained_backbone' : True,
        'backbone_depth' : 5,
        'num_classes' : num_classes,
        'dropout' : 0.2,
        'activation' : None,
        'coeff_channels' : [256,64],
        'num_splits' : int(crop_length // 5.0)
    },
    'optimizer':
        {
            'name' : 'Adam',
            'kwargs':{
                'lr' : learning_rate,
                'betas' : (0.9,0.99),
                'eps' : 1e-08,
                'weight_decay': 0.0001,
                'amsgrad': False
            }
        },
    'scheduler':
        {
            'name' : 'PolyLr',
            'kwargs':{
                'epochs' : epochs,
                'ratio' : 0.9,
            }
        },
    'loss' :
        {
            'name'  : loss_name,
            'kwargs' : {'logits':True,'smoothing':0.0}
        },

    
    'transform':{
        'level' : 2,
        'kwargs':{
            'nfft' : n_fft,
            'hop_length' : hop_length,
            'n_mels' : n_mels,
            'sample_rate': sampling_rate,
            'stype':'magnitude',
            'normalize' : True,
            'num_splits' : int(crop_length // 5.0)

        }
        
    },

    'dataset': dict(

        path='../../../datasets/birdclef-2022',
        fold = fold,
        crop_length = crop_length,#(seconds)
        sampling_rate = sampling_rate,
        n_fft = n_fft,
        hop_length = hop_length,
        totensor=False,
        pr_weight = 1.0,
        sec_weight = 1.0,
        neg_weight = 1.0,
        rating_min_weight = 0.0,
        use_ccw = False
    )
        ,
    'training':
        {   'activation' : 'sigmoid',
            'epochs' : epochs,
            'lr' : learning_rate,
            'batch_size' : 14,
            'val_batch_size' : 1,
            'accumulation_steps':4,
            'val_freq' : 1,
            'device' : 'cuda',
            'amp' : True,
            'n_classes' : num_classes,
            'max_grad_norm' : 1.0,
            'transform_level': 2,
            'use_mixup' : False,
            'use_coord_conv' : freq_coord_conv
        },


    'save_dir' : f'./results/bc22_{backbone}-{model}_{loss_name}_{epochs}epochs_01_fold{fold}_oversampled_tfmasks_2',

    'class_names' : class_names

}