import argparse
import os
import time

import torch.nn as nn
import torch.cuda
from torch.optim import Adam

from gat.constants import GATLayerType, GAT_CHECKPOINTS_PATH, GAT_BINARIES_PATH
from gat.definitions.gat import GAT

from utils.data_loading import load_graph_data
from constants import DatasetType, LoopPhase, ModelType

import utils.util as util
import utils.datasets_util as datasets_util
import utils.visualization as visualization




def get_main_loop(config, gat, cross_entropy_loss, optimizer, node_features, node_labels, adj_matrix, train_indices,
                  val_indices, test_indices, patience_period, time_start):
    node_dim = 0  # node axis

    train_labels = node_labels.index_select(node_dim, train_indices)
    val_labels = node_labels.index_select(node_dim, val_indices)
    test_labels = node_labels.index_select(node_dim, test_indices)

    # node_features shape = (N, FIN), edge_index shape = (2, E)
    graph_data = (node_features, adj_matrix)  # pack data into tuples because GAT uses nn.Sequential which requires it

    def get_node_indices(phase):
        if phase == LoopPhase.TRAIN:
            return train_indices
        elif phase == LoopPhase.VAL:
            return val_indices
        else:
            return test_indices

    def get_node_labels(phase):
        if phase == LoopPhase.TRAIN:
            return train_labels
        elif phase == LoopPhase.VAL:
            return val_labels
        else:
            return test_labels

    def main_loop(phase, epoch=0):
        global BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT, writer

        # Certain modules behave differently depending on whether we're training the model or not.
        # e.g. nn.Dropout - we only want to drop model weights during the training.
        if phase == LoopPhase.TRAIN:
            gat.train()
        else:
            gat.eval()

        node_indices = get_node_indices(phase)
        gt_node_labels = get_node_labels(phase)  # gt stands for ground truth

        # Do a forwards pass and extract only the relevant node scores (train/val or test ones)
        # Note: [0] just extracts the node_features part of the data (index 1 contains the edge_index)
        # shape = (N, C) where N is the number of nodes in the split (train/val/test) and C is the number of classes

        nodes_unnormalized_scores = gat(graph_data)[0].index_select(node_dim, node_indices)

        # In Cora dataset we will have 7 output probabilities. cross_entropy loss first applies softmax to vector,
        # and then it calculates loss
        # The probability of the correct class for most nodes approaches 1 we get to 0 loss!
        loss = cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)

        if phase == LoopPhase.TRAIN:
            optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
            loss.backward()  # compute the gradients for every trainable weight in the computational graph
            optimizer.step()  # apply the gradients to weights

        # Calculate the main metric - accuracy

        # Finds the index of maximum (unnormalized) score for every node and that's the class prediction for that node.
        # Compare those to true (ground truth) labels and find the fraction of correct predictions -> accuracy metric.
        class_predictions = torch.argmax(nodes_unnormalized_scores, dim=-1)
        accuracy = torch.sum(torch.eq(class_predictions, gt_node_labels).long()).item() / len(gt_node_labels)

        #
        # Logging
        #

        if phase == LoopPhase.TRAIN:
            # Log metrics
            if config['enable_tensorboard']:
                writer.add_scalar('training_loss', loss.item(), epoch)
                writer.add_scalar('training_acc', accuracy, epoch)

            # Save model checkpoint
            if config['checkpoint_freq'] is not None and (epoch + 1) % config['checkpoint_freq'] == 0:
                ckpt_model_name = f'gat_{config["dataset_name"]}_ckpt_epoch_{epoch + 1}.pth'
                config['test_perf'] = -1
                torch.save(util.get_gat_training_state(config, gat),
                           os.path.join(GAT_CHECKPOINTS_PATH, ckpt_model_name))

        elif phase == LoopPhase.VAL:
            # Log metrics
            if config['enable_tensorboard']:
                writer.add_scalar('val_loss', loss.item(), epoch)
                writer.add_scalar('val_acc', accuracy, epoch)

            # Log to console
            if config['console_log_freq'] is not None and epoch % config['console_log_freq'] == 0:
                print(
                    f'GAT training: time elapsed= {(time.time() - time_start):.2f} [s] | epoch={epoch + 1} | val acc={accuracy}')

            # The "patience" logic - should we break out from the training loop? If either validation acc keeps going up
            # or the val loss keeps going down we won't stop
            if accuracy > BEST_VAL_PERF or loss.item() < BEST_VAL_LOSS:
                BEST_VAL_PERF = max(accuracy, BEST_VAL_PERF)  # keep track of the best validation accuracy so far
                BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)  # and the minimal loss
                PATIENCE_CNT = 0  # reset the counter every time we encounter new best accuracy
            else:
                PATIENCE_CNT += 1  # otherwise keep counting

            if PATIENCE_CNT >= patience_period:
                raise Exception('Stopping the training.')

        else:
            return accuracy  # in the case of test phase we just report back the test accuracy

    return main_loop  # return the decorated function


def train_gat_cora(config):
    global BEST_VAL_PERF, BEST_VAL_LOSS

    # check whether you have GPU
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device( "cpu")

    # load data
    node_features, node_labels, adj_matrix = load_graph_data(config, device)

    if config["balanced"]:
        train_indices, val_indices, test_indices = \
            util.get_balanced_train_indices(node_labels.cpu().detach().numpy(),
                                            num_training_examples_per_class=datasets_util.get_num_training_examples_per_classes(
                                                config["dataset_name"]))
    else:
        train_indices, val_indices, test_indices = util.get_unbalanced_train_indices(
            *datasets_util.get_train_test_val_ranges(dataset_name = config["dataset_name"]))

    # Indices that help us extract nodes that belong to the train/val and test splits, currently hardcoded
    # question: what about training shuffle?
    train_indices = torch.as_tensor(train_indices, device=device)
    val_indices = torch.as_tensor(val_indices, device=device)
    test_indices = torch.as_tensor(test_indices, device=device)

    # Step 2: prepare the model
    gat = GAT(
        num_of_layers=config['num_of_layers'],
        num_features_per_layer=config['num_features_per_layer'],
        bias=config['bias'],
        dropout=config['dropout'],
        add_skip_connection=config['add_skip_connection'],
        num_heads_per_layer=config['num_heads_per_layer']
    ).to(device)

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    main_loop = get_main_loop(
        config,
        gat,
        loss_fn,
        optimizer,
        node_features,
        node_labels,
        adj_matrix,
        train_indices,
        val_indices,
        test_indices,
        config['patience_period'],
        time.time())

    BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT = [0, 0, 0]  # reset vars used for early stopping

    # Step 4: Start the training procedure
    for epoch in range(config['num_of_epochs']):
        # Training loop
        main_loop(phase=LoopPhase.TRAIN, epoch=epoch)

        # Validation loop
        with torch.no_grad():
            try:
                main_loop(phase=LoopPhase.VAL, epoch=epoch)
            except Exception as e:  # "patience has run out" exception :O
                print(str(e))
                break  # break out from the training loop

    # Step 5: Potentially test your model
    # Don't overfit to the test dataset - only when you've fine-tuned your model on the validation dataset should you
    # report your final loss and accuracy on the test dataset. Friends don't let friends overfit to the *train* data. <3
    if config['should_test']:
        test_acc = main_loop(phase=LoopPhase.TEST)
        config['test_perf'] = test_acc
        print(f'Test accuracy = {test_acc}')
    else:
        config['test_perf'] = -1

    # Save the latest GAT in the binaries directory
    torch.save(
        util.get_gat_training_state(config, gat),
        os.path.join(GAT_BINARIES_PATH,
                     util.get_available_binary_name(binary_dir=GAT_BINARIES_PATH, dataset_name=config['dataset_name'],
                                                    model_name=ModelType.GAT.name))
    )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_train", action='store_true', default=False, required=False)

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=1000)
    parser.add_argument("--patience_period", type=int,
                        help="number of epochs with no improvement on val before terminating", default=1000)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=5e-4)
    parser.add_argument("--should_test", action='store_true', default=True,
                        help='should test the model on the test dataset? (no by default)')
    parser.add_argument("--balanced", action='store_true', default=True,
                        help="should train with balanced num of labels")

    # Dataset related
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='dataset to use for training',
                        default=DatasetType.CORA.name)
    parser.add_argument("--should_visualize", action='store_true', help='should visualize the dataset? (no by default)')
    parser.add_argument("--model_name", type=str, required=False)

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", action='store_true', help="enable tensorboard logging (no by default)")
    parser.add_argument("--console_log_freq", type=int, help="log to output console (epoch) freq (None for no logging)",
                        default=100)
    parser.add_argument("--checkpoint_freq", type=int,
                        help="checkpoint model saving (epoch) freq (None for no logging)", default=1000)
    args = parser.parse_args()



    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    # Model architecture related
    gat_config = {
        "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "add_skip_connection": False,  # hurts perf on Cora
        "bias": True,  # result is not so sensitive to bias
        "dropout": 0.6,  # result is sensitive to dropout,
        "layer_type": GATLayerType.IMP1,
        "model_type": ModelType.GAT,
        "num_features_per_layer": [datasets_util.get_num_input_features(training_config["dataset_name"]), 64,
                                    datasets_util.get_num_classes(training_config["dataset_name"])],
        "num_heads_per_layer": [8, 1],
    }

    # Add additional config information
    training_config = {**gat_config, **training_config}

    print(training_config)

    return training_config


def main():
    args = get_args()

    if not args["no_train"]:
        train_gat_cora(args)

    if not args["should_visualize"]:
        return

    if not args["model_name"]:
        binary_name = util.get_last_binary_name(binary_dir=GAT_BINARIES_PATH, dataset_name=args['dataset_name'],
                                                model_name=ModelType.GAT.name)
        if binary_name is None:
            return
    else:
        binary_name = f"GAT_{args['dataset_name']}_{args['model_name']}.pth"

    visualization.visualize_gat(binary_name=binary_name, dataset_name=args['dataset_name'])


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
