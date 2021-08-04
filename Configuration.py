
config = {
    'Params': {
        'batch_size': 16,
        'epochs': 10,
        'lr': 8e-3,
        'wd': 1e-6,
        'alpha': [1],
        'prediction_timestep': 4,
        'GRU_hidden_size': 30,
        'z_len': 10,
        'plot_before_training': False,
        'focal_loss_gamma': 1
    },
    'Paths': {
        'data_folder': '/Neteera/Work/amichay.feldman/datasets/rest',
        'output_folder': '../runnings_output/Jan_17_#1'
    }

}