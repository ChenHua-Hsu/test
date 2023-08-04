import torch, os, sys
import numpy as np
import torch.nn.functional as F
sys.path.insert(1, '../')
import utils, argparse

def main():
    usage=''
    argparser = argparse.ArgumentParser(usage)
    argparser.add_argument('-s','--savefile',dest='savefilename', help='output filenme', default='test.pt', type=str)
    args = argparser.parse_args()
    savefilename = args.savefilename
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    filename = 'dataset_1_photons_1_tensor_no_pedding_euclidian_0.pt'
    loaded_file = torch.load(filename)
    showers = loaded_file[0]
    incident_energies = loaded_file[1]
    print(f'File {filename} contains:')
    print(f'{type(showers)} of {len(showers)} showers of type {type(showers[0])}')
    print(f'{type(incident_energies)} of {len(incident_energies)} incidient energies of type {type(incident_energies[0])}')
    
    max_nhits = 0
    custom_data = utils.cloud_dataset(filename, transform=utils.rescale_energies(), transform_y=utils.rescale_conditional(), device=device)
    #custom_data = utils.cloud_dataset(filename, device=device)

    # Have to homogenise with data padding before using dataloaders
    padded_showers = []
    rescaled_incident_e = []
    for shower in custom_data.data:
        if shower.shape[0] > max_nhits:
            max_nhits = shower.shape[0]
    shower_count = 0
    print(f'Maximum number of hits for all showers in file: {max_nhits}')
    for showers in custom_data.data:
        pad_hits = max_nhits-showers.shape[0]
        # Firstly, do any rescaling of the inputs
        rescaler = utils.rescale_energies()
        if showers.shape[0] == 0:
            print(f'incident e: {incident_energies[shower_count]} with {showers.shape[0]} hits')
            continue
        shower_data_transformed = rescaler(showers,incident_energies[shower_count])
        # Next, pad with sentinel value to make the showers uniform in length
        padded_shower = F.pad(input = shower_data_transformed, pad=(0,0,0,pad_hits), mode='constant', value=-20)
        padded_showers.append(padded_shower)
        # Rescale the conditional input for each shower
        rescaler_y = utils.rescale_conditional()
        incident_e_transformed = rescaler_y(incident_energies[shower_count],custom_data.min_y,custom_data.max_y)
        rescaled_incident_e.append(incident_e_transformed)
        shower_count+=1

    torch.save([padded_showers,torch.as_tensor(rescaled_incident_e)], savefilename)
    # Check
    padded_loaded_file = torch.load(savefilename)
    padded_showers = padded_loaded_file[0]
    padded_incident_e = padded_loaded_file[1]
    print(f'File {savefilename} contains:')
    print(f'{type(padded_showers)} of {len(padded_showers)} showers of type {type(padded_showers[0])}')
    print(f'{type(padded_incident_e)} of {len(padded_incident_e)} showers of type {type(padded_incident_e[0])}')

if __name__=='__main__':
    main()
    