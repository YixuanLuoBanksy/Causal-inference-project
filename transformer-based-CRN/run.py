from utils.cancer_simulation import get_cancer_sim_data
from utils.utils_functions import get_processed_data
import numpy as np
import torch
import os
import argparse
import logging
import torch
from torch.utils.data import Dataset
from models import * 
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim

# init random seed for duplication

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemo_coeff", default=2, type=int)
    parser.add_argument("--radio_coeff", default=2, type=int)
    parser.add_argument("--projection_horizon", default=5, type=int) 
    parser.add_argument("--b_load", default=False, type=bool)    
    parser.add_argument("--b_save", default=False, type=bool)    
    parser.add_argument("--results_dir", default='results', type=str)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--seed", default=0, type=int)




    return parser.parse_args()

def check_result_path(args):
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

def prepare_data(args):
    pickle_map = get_cancer_sim_data(chemo_coeff=args.chemo_coeff, radio_coeff=args.radio_coeff, b_load=args.b_load,
                                          b_save=args.b_save, model_root=args.results_dir)
    training_data = pickle_map['training_data']
    validation_data = pickle_map['validation_data']
    test_data = pickle_map['test_data']
    scaling_data = pickle_map['scaling_data']

    training_processed = get_processed_data(training_data, scaling_data)
    validation_processed = get_processed_data(validation_data, scaling_data)
    test_processed = get_processed_data(test_data, scaling_data)

    return training_processed, validation_processed, test_processed                           

class CausalDataset(Dataset):
    def __init__(self, input_dataset):
        self.current_treatment = torch.tensor(input_dataset['current_treatments'])
        self.previous_treatment = torch.tensor(input_dataset['previous_treatments'])
        self.covariate = torch.tensor(input_dataset['current_covariates'])
        self.output = torch.tensor(input_dataset['outputs'])
        self.active_entries = torch.tensor(input_dataset['active_entries'])
        self.max_length = self.current_treatment.shape[1]
        self.init_data()

    def init_data(self):
        self.previous_treatment = self.init_previous_treatment(self.previous_treatment)
        self.previous_output = self.init_previous_output(self.output)
        self.previous_covariate = self.init_previous_covariate(self.covariate)
        self.init_input_vector()
    
    def init_previous_treatment(self, previous_treatment):
        all_zeros = torch.zeros((previous_treatment.shape[0], 1, previous_treatment.shape[2]))
        previous_treatment = torch.cat((all_zeros, previous_treatment),axis=1)
        return previous_treatment
    
    def init_previous_output(self, previous_output):
        all_zeros = torch.zeros((previous_output.shape[0], 1, previous_output.shape[2]))
        previous_output = torch.cat((all_zeros, previous_output[:,:-1,:]),axis=1)
        return previous_output
    
    def init_previous_covariate(self, previous_covariate):
        all_zeros = torch.zeros((previous_covariate.shape[0], 1, previous_covariate.shape[2]))
        previous_covariate = torch.cat((all_zeros, previous_covariate[:,:-1,:]),axis=1)
        return previous_covariate
    
    def init_input_vector(self):
        self.input_vector = torch.cat((self.previous_treatment, self.previous_output, self.previous_covariate), axis=-1)

    def __len__(self):
        return self.input_vector.shape[0]

    def __getitem__(self, idx):
        input_vector = self.input_vector[idx]
        current_outcome = self.output[idx]
        current_treatment = self.current_treatment[idx]
        current_covariate = self.covariate[idx]
        active_entries = self.active_entries[idx]
        return input_vector, current_outcome, current_treatment, current_covariate, active_entries


if __name__ == '__main__':

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    args = init_arg()
    check_result_path(args)

    logging.info("Random seed is {}".format(args.seed))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    '''
    keys: 
    dict_keys(['cancer_volume', 'chemo_dosage', 'radio_dosage', 'chemo_application', 'radio_application', 'chemo_probabilities', 'radio_probabilities', 
    'sequence_lengths', 'patient_types', 'current_covariates', 'previous_treatments', 'current_treatments', 'outputs', 'active_entries', 
    'unscaled_outputs', 'input_means', 'inputs_stds', 'output_means', 'output_stds'])
    '''
    training_processed, validation_processed, test_processed  = prepare_data(args)

    train_dataset = CausalDataset(training_processed)
    val_dataset = CausalDataset(validation_processed)
    test_dataset = CausalDataset(test_processed)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    if args.device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.MSELoss()
    # training components -> optimizer / dataset / scheduler / model
    train_loss_array = []
    val_loss_array = []

    logging.info("Training start")
    for epoch in tqdm(range(args.epochs)):
        train_loss = 0
        model.train()
        for input_vector, current_outcome, current_treatment, current_covariate, active_entries in train_dataloader:

            input_vector = input_vector.to(device).float()
            current_outcome = current_outcome.to(device).float()
            current_treatment = current_treatment.to(device).float()
            current_covariate = current_covariate.to(device).float()
            active_entries = active_entries.to(device).float()

            y_prediction, x_prediction = model(input_vector, current_covariate, current_treatment)

            loss = torch.pow(y_prediction - current_outcome, 2).mul(active_entries).mean() + torch.pow(x_prediction - current_treatment, 2).mul(active_entries).mean()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        scheduler.step()
        train_loss /= len(train_dataset)
        print("Epoch: {} | Training Loss: {}".format(epoch, train_loss))
        
        model.eval()
        val_loss = 0
        for input_vector, current_outcome, current_treatment, current_covariate, active_entries in train_dataloader:
            
            input_vector = input_vector.to(device).float()
            current_outcome = current_outcome.to(device).float()
            current_treatment = current_treatment.to(device).float()
            current_covariate = current_covariate.to(device).float()
            active_entries = active_entries.to(device).float()
            with torch.no_grad():
                y_prediction, x_prediction = model(input_vector, current_covariate, current_treatment)
            loss = torch.sqrt(torch.pow(y_prediction - current_outcome, 2).mul(active_entries).mean()) / 1150
       
            val_loss += loss.item()
        print("Epoch: {} | Val Loss: {}".format(epoch, val_loss))






