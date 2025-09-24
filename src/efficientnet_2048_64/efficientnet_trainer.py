
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from torch import nn
import timm
#from torchvision import transforms
from torchvision import transforms, tv_tensors
from torchvision.transforms import v2
from src.utils.arguments import get_args
# from asuka.modeling.mae_module import MAE_Module
from src.utils.wandb import create_trainer, load_from_disk_, save_checkpoint
from src.base_ptln import BasePTLN
from src.efficientnet_2048_64.efficientnet_pipeline import EfficientNetPipeline
from src.efficientnet_2048_64.efficientnet_ptln import EfficientNetPTLN
import logging
import torch
import torch.multiprocessing as mp
from torchvision.models import efficientnet_b1, inception_v3
from src.base_dataset import BaselineDataset, BalancedGeneratorSampler, BalancedGeneratorSampler_Stage2
from src.efficientnet_2048_64.efficientnet_dataset import EfficientNetDataset

mp.set_start_method("spawn", force=True)
logger = logging.getLogger(__name__)
def main(args):
    trainer, device, dist = create_trainer(args)
    if args.do_train == False and args.do_train_stage2 == False and args.do_eval == False:
        raise ValueError("At least one of `do_train`, 'do_train_stage2' or `do_eval` must be True.")

    train_streaming = True
    if args.do_train:
        json_file = args.json_file

        transformation = v2.Compose([
            v2.Resize((args.resolution, args.resolution)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=10),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
            
            v2.ToImage(),               # makes it a torch tensor
            v2.ToDtype(torch.float32, scale=True),  # cast to float32
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        inplace=True),
        ])

        training_dataset = EfficientNetDataset(json_file=json_file, transformation=transformation)

        samples_per_label = {'efs': 32, 'fr': 32, 'fs': 32, 'real': 32}  
        sampler = BalancedGeneratorSampler(training_dataset, samples_per_label)
                
        training_dataloader = torch.utils.data.DataLoader(
            training_dataset,
            batch_sampler=sampler,
            num_workers=args.dataloader_num_workers,
            shuffle= not train_streaming,
            pin_memory=True,
            persistent_workers=True,
            #prefetch_factor=2
        )
        del training_dataset
        del sampler
        del json_file

    if args.do_train_stage2:
        #train_name = 'data/dataset_stage1.pt'
        json_file = args.json_file

        transformation = v2.Compose([
            v2.Resize((args.resolution, args.resolution)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=10),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
            
            v2.ToImage(),               # makes it a torch tensor
            v2.ToDtype(torch.float32, scale=True),  # cast to float32
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        inplace=True),
        ])

        training_dataset = EfficientNetDataset(json_file=json_file, transformation=transformation)

        samples_per_generator = 2  
        sampler = BalancedGeneratorSampler_Stage2(training_dataset, samples_per_generator)
                
        training_dataloader = torch.utils.data.DataLoader(
            training_dataset,
            batch_sampler=sampler,
            num_workers=args.dataloader_num_workers,
            shuffle= not train_streaming,
            pin_memory=True,
            persistent_workers=True,

        )

    if args.do_eval:
        json_file = 'data/image_labels_test.json'
        feature_extractor = timm.create_model('resnet50_clip.openai', pretrained=True)
        transformation = transforms.Compose([
            transforms.Resize((args.resolution,args.resolution)),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        eval_dataset = EfficientNetDataset(json_file=json_file, model = feature_extractor,transformation=transformation)
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.val_batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle = None
        )
    

    # delete to free memory
    """
    del  vae_moduel
    """
    # del ctdn #IF
    torch.cuda.empty_cache()
    
    if args.do_train:
        logger.info("*** Load Inception module ***")
        #model = inception_v3(pretrained=True, aux_logits=True).to(device)
        model = efficientnet_b1(pretrained=True)
        model.train()
        feature_extractor = timm.create_model('resnet50_clip.openai', pretrained=True)
        feature_extractor.eval()
        logger.info("*** Load pipeline module ***")
        pipeline = EfficientNetPipeline(model = model, device=device, in_features=3, num_label=4)
        pipeline.train()
       # for name, param in asuka_pipeline_xl.named_parameters():
        #    print(f"{name}: {param.numel()}")

        logger.info("*** Load trainer ***")
        try:
            trainer.fit(
                model=EfficientNetPTLN(args=args, pipeline=pipeline, sync_dist=dist, feature_extractor=feature_extractor),
                train_dataloaders=training_dataloader,
                #val_dataloaders=eval_dataloader if args.do_eval else None,
            )
        except KeyboardInterrupt:
            logger.warning("Training interupted by user")
            save_checkpoint(trainer=trainer, args=args)
        else:
            save_checkpoint(trainer=trainer, args=args)

    if args.do_train_stage2:
        ckpt_path = f'checkpoint/Deepfake_Detection_512_256_stage1/checkpoint/best.pt'
        checkpoint = torch.load(ckpt_path, map_location=device)
        feature_extractor = timm.create_model('resnet50_clip.openai', pretrained=True)
        feature_extractor.eval()

        # Recreate the model architecture
        model = efficientnet_b1(pretrained=False)  # Don't load ImageNet weights
        model.train()
        pipeline = EfficientNetPipeline( 
            model=model, 
            device=device, 
            in_features=3, 
            num_label=4  # Original number of classes from training
        )
        
        # Create Lightning module
        trained_model = EfficientNetPTLN(args=args, pipeline=pipeline, sync_dist=dist, feature_extractor=feature_extractor)
        
        # Load state dict
        trained_model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("Successfully loaded manually")
    
        # Modify classifier for downstream task
        pipeline = trained_model.pipeline
        new_num_classes = args.num_label 
        model_input_dim = pipeline.model.classifier[1].in_features
        pipeline.model.classifier[1] = nn.Linear(model_input_dim, new_num_classes).to(device)

        logger.info(f"Training for {args.num_train_epochs} epochs")
        logger.info(f"Model device: {next(trained_model.parameters()).device}")
        logger.info(f"New classifier: {pipeline.model.classifier[1]}")
        print("Trainer thinks current_epoch =", trainer.current_epoch)
        print("Trainer max_epochs =", trainer.fit_loop.max_epochs)

        # Check if validation dataloader is working
        if args.do_eval and eval_dataloader:
            logger.info(f"Validation dataloader length: {len(eval_dataloader)}")
        
        logger.info("*** Load trainer ***")
        trainer, _, _ = create_trainer(args) # do this to create new trainer when train 2nd stage (yes bad code practice, i know)
        try:
            trainer.fit(
                model=trained_model,
                train_dataloaders=training_dataloader,
                val_dataloaders=eval_dataloader if args.do_eval else None,
            )
        except KeyboardInterrupt:
            logger.warning("Training interupted by user")
            save_checkpoint(trainer=trainer, args=args)
        else:
            save_checkpoint(trainer=trainer, args=args)
        
    # if args.do_test:
    #     logger.info("\n\n*** Evaluate ***")
    #     trainer.devices = 0
    #     trainer.test(
    #         asuka_pipeline_xl(args, dit_module=dit_module, vae=vae_m,noise_scheduler=noise_scheduler,  align_module=align_module),
    #         dataloaders=eval_dataloader,
    #         ckpt_path="best"
    #     )

if __name__ == "__main__":
    opt = get_args()
    
    logger.info("*** Training mode ***")
    main(opt)