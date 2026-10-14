print('Importing libraries...')
# Standard library imports
import argparse
import time
import json
from pathlib import Path
import warnings 

# Third-party imports
import torch
import numpy as np
import torch.nn as nn

# Local imports
from scripts.f_environment import get_config, set_deterministic_behaviour, verify_results_weights_folder
from scripts.f_dataset import get_datasets, get_dataloaders
from scripts.f_build import build_model
from scripts.f_training_utils import build_optimizer, update_params, NativeScalerWithGradNormCount
from scripts.f_metrics import get_map, get_balanced_accuracies
from scripts.f_training import save_weights
warnings.filterwarnings("ignore")

##############################################################################################
##############################################################################################
# ENVIRONMENT
pwd = Path.cwd()
print(f"Current working directory: {pwd}")

# Verify necessary folder structure and download weights
verify_results_weights_folder(pwd)

# Load config
parser = argparse.ArgumentParser(description="Run SwinCVS with specified config")
parser.add_argument('--config_path', type=str, required=True, help='Path to config YAML file')
args = parser.parse_args()
config_path = args.config_path
config, experiment_name = get_config(config_path)

seed = config.SEED
set_deterministic_behaviour(seed)


##############################################################################################
##############################################################################################
# DATASET and DATALOADER
training_dataset, val_dataset, test_dataset = get_datasets(config)
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config, training_dataset, val_dataset, test_dataset)

##############################################################################################
##############################################################################################

# Initialise SwinCVS according to config
model = None
model = build_model(config)
print('Full model initialised successfully!\n')

# Load saved weights for inference
if config.MODEL.INFERENCE:
    weights = "weights/" + config.MODEL.INFERENCE_WEIGHTS
    model.load_state_dict(torch.load(weights))
    print(f"Trained SwinCVS weights loaded successfully for INFERENCE - name: {config.MODEL.INFERENCE_WEIGHTS}")
model.to('cuda')
torch.cuda.empty_cache()

##############################################################################################
##############################################################################################
optimizer = build_optimizer(config, model)
loss_scaler = NativeScalerWithGradNormCount()
class_weights = torch.tensor(config.TRAIN.CLASS_WEIGHTS).to('cuda')
criterion = nn.BCEWithLogitsLoss(weight=class_weights).to('cuda')

##############################################################################################
##############################################################################################
# TRAINING #
# Training variables
print(f"Experiment name: {experiment_name}\n")
results_dict = {}
if not config.MODEL.INFERENCE:
    num_epochs = config.TRAIN.EPOCHS
    checkpoint_path = pwd / 'weights'
    best_mAP = 0
    start_time = 0
    end_time = 0
    time_list = []

    if config.MODEL.MULTICLASSIFIER:
        multiclasifier_alpha = config.TRAIN.MULTICLASSIFIER_ALPHA
        multiclasifier_beta = 1-multiclasifier_alpha

    print("Beginning training...")
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1:02}/{num_epochs:02}")

        # Update weight scaling parameters if 
        if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
            multiclasifier_alpha, multiclasifier_beta = update_params(multiclasifier_alpha, multiclasifier_beta, epoch)
        
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        optimizer.zero_grad()

        print("Training")
        start_time = time.time()
        for idx, (samples, targets) in enumerate(train_dataloader):
            print(f"Processing batch: {idx+1:04}/{len(train_dataloader):04}", end="\r")

            # Get predictions
            samples, targets = samples.to('cuda'), targets.to('cuda')
            with torch.amp.autocast("cuda", enabled=True):
                if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
                    outputs_swin, outputs_lstm = model(samples)
                else:
                    outputs_lstm = model(samples)

            # Get loss and backpropagation
            if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
                loss_train = multiclasifier_alpha*criterion(outputs_swin, targets) + multiclasifier_beta*criterion(outputs_lstm, targets)
            else:
                loss_train = criterion(outputs_lstm, targets)

            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss_train, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
            optimizer.zero_grad()
            train_loss+=loss_train.item()
            torch.cuda.synchronize()
        
        # Validation Epochs
        print("\nValidation")
        model.eval()
        val_probabilities = []
        val_predictions = []
        val_targets = []
        with torch.inference_mode():
            for idx, (samples, targets) in enumerate(val_dataloader):
                print(f"Processing batch: {idx+1:04}/{len(val_dataloader):04}", end="\r")
                # Get predictions
                samples, targets = samples.to('cuda'), targets.to('cuda')
                if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
                    outputs_swin, outputs_lstm = model(samples)
                else:
                    outputs_lstm = model(samples)
                
                # Get outputs
                val_probability = torch.sigmoid(outputs_lstm)
                val_prediction = torch.round(val_probability)

                # Save outputs
                val_probabilities.append(val_probability.to('cpu'))
                val_predictions.append(val_prediction.to('cpu'))
                val_targets.append(targets.to('cpu'))

                # Loss
                loss_val = criterion(outputs_lstm, targets)
                val_loss += loss_val.item()
                torch.cuda.synchronize()

        # Get validation scores
        C1_balanced_accuracy, C2_balanced_accuracy, C3_balanced_accuracy, total_balanced_accuracy = get_balanced_accuracies(val_targets, val_predictions)
        C1_ap, C2_ap, C3_ap, mAP = get_map(val_targets, val_probabilities)
        print('\nAverage balanced accuracy', round((C1_balanced_accuracy+C2_balanced_accuracy+C3_balanced_accuracy)/3, 4))
        print('C1 bacc', round(C1_balanced_accuracy, 4))
        print('C2 bacc', round(C2_balanced_accuracy, 4))
        print('C3 bacc', round(C3_balanced_accuracy, 4))
        print('mAP', round(mAP, 4))
        print('C1 ap', round(C1_ap, 4))
        print('C2 ap', round(C2_ap, 4))
        print('C3 ap', round(C3_ap, 4))

        # Save validation scores
        val_predictions_2save = torch.cat(val_predictions, dim=0).tolist()
        val_probabilities_2save = torch.cat(val_probabilities, dim=0).tolist()
        val_targets_2save = torch.cat(val_targets, dim=0).tolist()

        epoch_results = {'avg_bal_acc': round(total_balanced_accuracy, 4),
                        'C1_bacc': round(C1_balanced_accuracy, 4), 'C2_bacc': round(C2_balanced_accuracy, 4), 'C3_bacc': round(C3_balanced_accuracy, 4),
                        'avg_map': round(mAP, 4),
                        'C1_map': round(C1_ap, 4), 'C2_map': round(C2_ap, 4), 'C3_map': round(C3_ap, 4),
                        'preds': val_predictions_2save, 'true': val_targets_2save, 'preds_prob': val_probabilities_2save,
                        'train_loss': train_loss, 'val_loss': val_loss}
        results_dict[f"Epoch {epoch+1}"] = epoch_results

        # Save results
        with open(pwd / 'results' / f'{experiment_name}_results.json', 'w') as file:
            json.dump(results_dict, file, indent=4)

        # Estimate remaining time
        end_time = time.time()
        time_of_epoch = int(end_time-start_time)
        print(F"Epoch duration: {time_of_epoch}s")
        time_list.append(time_of_epoch)
        print(f"Estimated remaining time: {np.round(np.mean(time_list)*(num_epochs-(epoch+1)))}s")
        
        # Save weights of the best epoch
        if mAP >= best_mAP:
            best_mAP = mAP
            print(f"New best result (Epoch {epoch+1}), saving weights...")
            save_weights(model, config, experiment_name)
        else:
            print('\n')



######
# ADD CHOOSING AND LOADING BEST EPOCH FROM TRAINIG IF NOT INFERENCE
best_epoch = 0

# Test time measurement variables
start_time = 0
end_time = 0
times = []

# Performance measurement variables
test_probabilities = []
test_predictions = []
test_targets = []

len_dataloader = len(test_dataloader)
print('\nTesting')
model.eval()
with torch.inference_mode():
    for idx, (samples, targets) in enumerate(test_dataloader):
        print(f"Processing batch: {idx+1:04}/{len_dataloader:04}", end="\r")

        # Time start
        start_time = time.time()

        # Get preds
        samples, targets = samples.to('cuda'), targets.to('cuda')
        
        if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER and not config.MODEL.INFERENCE:
            outputs_swin, outputs_lstm = model(samples)
        else:
            outputs_lstm = model(samples)

        # Get outputs
        test_probability = torch.sigmoid(outputs_lstm)
        test_prediction = torch.round(test_probability)

        # Time end
        end_time = time.time()
        elapsed_time  = end_time-start_time
        times.append(elapsed_time)

        # Save results from a batch to a list
        test_probabilities.append(test_probability.to('cpu'))
        test_predictions.append(test_prediction.to('cpu'))
        test_targets.append(targets.to('cpu'))

        torch.cuda.synchronize()

# Calculate metrics
C1_balanced_accuracy, C2_balanced_accuracy, C3_balanced_accuracy, total_balanced_accuracy = get_balanced_accuracies(test_targets, test_predictions)
C1_ap, C2_ap, C3_ap, mAP = get_map(test_targets, test_probabilities)

# Print metrics
print('\nTesting results:')
print('Average balanced accuracy', round((C1_balanced_accuracy+C2_balanced_accuracy+C3_balanced_accuracy)/3, 4))
print('C1 bacc', round(C1_balanced_accuracy, 4))
print('C2 bacc', round(C2_balanced_accuracy, 4))
print('C3 bacc', round(C3_balanced_accuracy, 4))
print('mAP', round(mAP, 4))
print('C1 ap', round(C1_ap, 4))
print('C2 ap', round(C2_ap, 4))
print('C3 ap', round(C3_ap, 4))

mean_inference = round(np.mean(times)*1000,1)
std_inference = round(np.std(times)*1000,1)
total_inference = round(np.sum(times),1)
print(f"Inference time: mean={mean_inference}ms, std={std_inference}ms, total={total_inference}s")

test_predictions_2save = torch.cat(test_predictions, dim=0).tolist()
test_probabilities_2save = torch.cat(test_probabilities, dim=0).tolist()
test_targets_2save = torch.cat(test_targets, dim=0).tolist()

epoch_results = {'avg_bal_acc': round(total_balanced_accuracy, 4),
                'C1_bacc': round(C1_balanced_accuracy, 4), 'C2_bacc': round(C2_balanced_accuracy, 4), 'C3_bacc': round(C3_balanced_accuracy, 4),
                'avg_map': round(mAP, 4),
                'C1_map': round(C1_ap, 4), 'C2_map': round(C2_ap, 4), 'C3_map': round(C3_ap, 4),
                'preds': test_predictions_2save, 'true': test_targets_2save, 'preds_prob': test_probabilities_2save,
                'mean_inference_ms': mean_inference, 'std_inference_ms': std_inference, 'total_inference_s': total_inference}
results_dict[f"Testing_@E{best_epoch+1}"] = epoch_results # CHANGE THE BEST EPOCH

with open(pwd / 'results' / f'{experiment_name}_results.json', 'w') as file:
    json.dump(results_dict, file, indent=4)