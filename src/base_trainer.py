
from torch import nn
from torchvision import transforms
from utils.arguments import get_args
# from asuka.modeling.mae_module import MAE_Module
from utils.wandb import create_trainer, load_from_disk_, save_checkpoint
from base_ptln import BasePTLN
from base_pipeline import BasePipeline
import logging
import torch
import os
from torchvision.models import efficientnet_b1, inception_v3
from base_dataset import BaselineDataset, BalancedGeneratorSampler, BalancedGeneratorSampler_Stage2

logger = logging.getLogger(__name__)
def main(args):
    trainer, device, dist = create_trainer(args)
    if args.do_train == False and args.do_train_stage2 == False and args.do_eval == False:
        raise ValueError("At least one of `do_train`, 'do_train_stage2' or `do_eval` must be True.")
    # logger.info("*** Load VAE module ***")
    # vae_module = AutoencoderKL.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     subfolder="vae",
    #     revision=args.revision,
    #     variant=args.variant
    # ).to(device)

    train_streaming = True
    #dataset_ = dataset_inpaint(args=args, vae_module=vae_module)
    if args.do_train:
        #train_name = 'data/dataset_stage1.pt'
        json_file = 'data/image_labels.json'

        transformation = transforms.Compose([
            transforms.Resize((args.resolution,args.resolution)),

            transforms.RandomApply([
                transforms.RandomHorizontalFlip(p=1.0)
            ], p=0.5),  # flip prob

            transforms.RandomApply([
                transforms.RandomRotation(degrees=10)
            ], p=0.5),  # rotate prob, limit [-10, 10]

            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.1)
            ], p=0.5),  # brightness prob

            transforms.RandomApply([
                transforms.ColorJitter(contrast=0.1)
            ], p=0.5),  # contrast prob

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        training_dataset = BaselineDataset(json_file=json_file, model=None, transformation=transformation)

        samples_per_label = {'efs': 32, 'fr': 32, 'fs': 32, 'real': 32}  
        sampler = BalancedGeneratorSampler(training_dataset, samples_per_label)
                
        training_dataloader = torch.utils.data.DataLoader(
            training_dataset,
            batch_sampler=sampler,
            num_workers=args.dataloader_num_workers,
            shuffle= not train_streaming,
            pin_memory=True,
            persistent_workers=True,

        )
        del training_dataset
        del sampler
        del json_file

    if args.do_train_stage2:
        #train_name = 'data/dataset_stage1.pt'
        json_file = 'data/image_labels_stage2.json'

        transformation = transforms.Compose([
            transforms.Resize((args.resolution,args.resolution)),

            transforms.RandomApply([
                transforms.RandomHorizontalFlip(p=1.0)
            ], p=0.5), 

            transforms.RandomApply([
                transforms.RandomRotation(degrees=10)
            ], p=0.5), 

            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.1)
            ], p=0.5), 

            transforms.RandomApply([
                transforms.ColorJitter(contrast=0.1)
            ], p=0.5), 

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        training_dataset = BaselineDataset(json_file=json_file, transformation=transformation)

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
        json_file = 'data/image_labels_stage2.json'

        transformation = transforms.Compose([
            transforms.Resize((args.resolution,args.resolution)),

            transforms.RandomApply([
                transforms.RandomHorizontalFlip(p=1.0)
            ], p=0.5), 

            transforms.RandomApply([
                transforms.RandomRotation(degrees=10)
            ], p=0.5), 

            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.1)
            ], p=0.5), 

            transforms.RandomApply([
                transforms.ColorJitter(contrast=0.1)
            ], p=0.5), 

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        eval_dataset = BaselineDataset(json_file=json_file, transformation=transformation)
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
        logger.info("*** Load pipeline module ***")
        pipeline = BasePipeline(model = model, device=device, in_features=3, num_label=4)
        pipeline.train()
       # for name, param in asuka_pipeline_xl.named_parameters():
        #    print(f"{name}: {param.numel()}")

        logger.info("*** Load trainer ***")
        try:
            trainer.fit(
                model=BasePTLN(args=args, pipeline=pipeline, sync_dist=dist),
                train_dataloaders=training_dataloader,
                #val_dataloaders=eval_dataloader if args.do_eval else None,
            )
        except KeyboardInterrupt:
            logger.warning("Training interupted by user")
            save_checkpoint(trainer=trainer, args=args)
        else:
            save_checkpoint(trainer=trainer, args=args)

    if args.do_train_stage2:
        ckpt_path = f'checkpoint/Deepfake_Detection_no_encoder_stage1/checkpoint/best.pt'
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Recreate the model architecture
        model = efficientnet_b1(pretrained=False)  # Don't load ImageNet weights
        model.train()
        pipeline = BasePipeline( 
            model=model, 
            device=device, 
            in_features=3, 
            num_label=4  # Original number of classes from training
        )
        
        # Create Lightning module
        trained_model = BasePTLN(args=args, pipeline=pipeline, sync_dist=dist)
        
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