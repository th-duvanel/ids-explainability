import flwr as fl
from dnn_model import create_model
import argparse
from utils import load_config

def get_on_fit_config_fn(batch_size, local_epochs):
    ''' Returns a functions which returns the server config file,
    to be used by the clients
    '''
    def fit_config(server_round):
        
        config = {
            "batch_size": batch_size,
            "current_round": server_round,
            "local_epochs": local_epochs,
        }
        
        return config
    
    return fit_config

def parse_arguments():
    '''Parser
    '''
    parser = argparse.ArgumentParser(description='FL server')
    parser.add_argument('--config', type=str, default='conf/server/config.json', help='Path to the config file: conf/server/config.json')
    parser.add_argument('--server_address', type=str, help='Address of server')
    parser.add_argument('--num_rounds', type=int, help='Number of rounds')
    parser.add_argument('--num_clients', type=int, help='Number of clients')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--local_epochs', type=int, help='Number of local epochs')
    parser.add_argument('--input_shape', type=int, help='Model input shape')
    parser.add_argument('--num_classes', type=int, help='Number of classes')
    parser.add_argument('--loss', type=str, help='Loss function of the model')
    
    return parser.parse_args()

def update_config(args, conf):
    '''Update config file via cli
    '''
    if args.server_address is not None:
        conf['server_address'] = args.server_address
    if args.num_rounds is not None:
        conf['num_rounds'] = args.num_rounds
    if args.num_clients is not None:
        conf['num_clients'] = args.num_clients
    if args.batch_size is not None:
        conf['batch_size'] = args.batch_size
    if args.local_epochs is not None:
        conf['local_epochs'] = args.local_epochs
    if args.input_shape is not None:
        conf['input_shape'] = args.input_shape
    if args.num_classes is not None:
        conf['num_classes'] = args.num_classes
    if args.loss is not None:
        conf['loss'] = args.loss
    
    return conf
    

def main():
    
    # parser
    args = parse_arguments()
    
    # load config
    conf = load_config(args.config)
    conf = update_config(args, conf)
    
    # model architecture
    model = create_model(conf['input_shape'], conf['num_classes'])
    
    # compile model
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # get model weights
    weights = model.get_weights()
    
    # Serialize ndarrays to "Parameters"
    parameters = fl.common.ndarrays_to_parameters(weights)

    # Server's strategy
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=conf["num_clients"],
        on_fit_config_fn=get_on_fit_config_fn(conf["batch_size"], conf['local_epochs']),
        initial_parameters=parameters,
    )
    
    # Start server
    fl.server.start_server(
        server_address=conf['server_address'],
        config=fl.server.ServerConfig(num_rounds=conf['num_rounds']),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()