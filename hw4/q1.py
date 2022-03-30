import argparse 
import numpy as np
import pandas as pd



def split_data(data, labels, num_train):
    """
    Spit into training and testing set
    """
    train_data = data[:num_train]
    train_lbls = labels[:num_train]

    test_data = data[2000:]
    test_lbls = labels[2000:]

    return train_data, train_lbls, test_data, test_lbls


def get_data(num_train):
    """
    Split the data and and labels into training and testing sets 
    """
    data = pd.read_csv("spam_data.txt", delimiter='\s+', header=None)
    data[57] = 1  # Add intercept term
    data = data.to_numpy()

    labels = pd.read_csv("spam_labels.txt", delimiter='\s+', header=None)
    labels = labels.to_numpy().squeeze()

    # Convert to 1/-1 encoding
    labels = np.where(labels == 0, -1, 1)

    return split_data(data, labels, num_train)


def sigmoid(x):
    """
    Apply the sigmoid function to input
    """
    return 1 / (1 + np.exp(-x))


def get_preds(data, weights):
    """
    Get predictions
    """
    return sigmoid(data @ weights)


def bce(preds, lbls):
    """
    Calculate binary cross entropy loss on given set of sample
    """
    return np.log(1 + np.exp(-lbls * preds)).mean()
    

def gradient(data, labels, weights):
    """
    Calculate the gradient for our single prediction
    """
    total_grad = np.zeros(data.shape[1], dtype=np.float64)

    for i in range(data.shape[0]):
        denom = 1 + np.exp(labels[i] * np.dot(data[i], weights))
        single_grad = labels[i] * data[i] / denom
        total_grad += single_grad

    return - total_grad / data.shape[0]


def abs_diff(iter_preds, prev_iter_preds):
    """
    Calculate the absolute difference in predictions between the current
    iteration and the previous.

    This is compared to the convergence criteria (epsilon)
    """
    diff = iter_preds - prev_iter_preds
    return np.sum(np.abs(diff))



def logistic_train(data, labels, epsilon=1e-5, maxiter=1000, lr=1):
    """
    Train logistic regression
    """
    prev_iter_preds = []
    weights = np.zeros(data.shape[1]) 

    for iter in range(1, maxiter+1):
        preds = get_preds(data, weights)

        loss = bce(preds, labels)
        grad = gradient(data, labels, weights)
        weights = weights + lr * grad

        print(f"Iter {iter} Mean Loss:", loss)

        # Check if reached convergence. If so stop training
        if iter > 1:
            if abs_diff(preds, prev_iter_preds) < epsilon:
                print(f"Reached convergence at iteration {iter}. Stopping training!")
                break
                
        prev_iter_preds = preds

    return weights 


def eval_model(weights, test_data, test_lbls):
    """
    Eval model on test set
    """
    preds = test_data @ weights
    preds = np.where(preds > 0, 1, -1)

    correct_preds = (preds == test_lbls).sum()
    acc = correct_preds / test_data.shape[0] * 100

    print(f"Accuracy: {acc:.2f}%")



def main():
    parser = argparse.ArgumentParser(description='Logistic Regression parameters')
    
    parser.add_argument('--lr', help="Learning rate", type=float, default=1)
    parser.add_argument('--maxiter', help='Max number of iterations to run', type=int, default=1000)
    parser.add_argument('--epsilon', help='Convegence criteria', type=float, default=1e-5)
    parser.add_argument('--train-on', help='Train on the first n samples', type=int, default=2000)

    args = parser.parse_args()

    train_data, train_lbls, test_data, test_lbls = get_data(args.train_on)

    print(train_lbls.shape, train_lbls[train_lbls == 1].shape)

    weights = logistic_train(train_data, train_lbls, epsilon=args.epsilon, maxiter=args.maxiter, lr=args.lr)

    eval_model(weights, test_data, test_lbls)

    
if __name__ == "__main__":
    main()
