import numpy as np
import neuralnet_starter
import pickle


def main():

    config = {}
    config['layer_specs'] = [784,100,10]
    config['batch_size'] = 1
    config['activation'] = 'sigmoid'
    config['epochs'] = 1
    config['early_stop'] = True  # Implement early stopping or not
    config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
    config['L2_penalty'] = 0  # Regularization constant
    config['momentum'] = False  # Denotes if momentum is to be applied or not
    config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression


    np.random.seed(42)
    train_images,train_targets = neuralnet_starter.load_data("MNIST_train.pkl")
    x_image = np.array([train_images[0]])
    targets = train_targets[0]
    nnet = neuralnet_starter.Neuralnetwork(config)
    nnet.forward_pass(x_image,targets)
    nnet.backward_pass() #Has the gradients computed by backprop

    #Do numerical gradient computation

    #Output bias weight
    outputLayer = nnet.layers[-1]
    hiddenLayer = nnet.layers[0]

    epsilon = 1e-2
    #Output units
    outputBiasGradient = compute_gradient(x_image,targets,nnet,outputLayer,"bias",[0,0],epsilon)
    check_error(outputBiasGradient,-outputLayer.d_b[0,0],epsilon**2,"Output Bias Gradients")
    outputWeight1Gradient = compute_gradient(x_image,targets,nnet,outputLayer,"weights",[5,4],epsilon)
    check_error(outputWeight1Gradient,-outputLayer.d_w[5,4],epsilon**2,"Output Weight 1 Gradients")
    outputWeight2Gradient = compute_gradient(x_image,targets,nnet,outputLayer,"weights",[90,3],epsilon)
    check_error(outputWeight2Gradient,-outputLayer.d_w[90,3],epsilon**2,"Output Weight 2 Gradients")

    #Hidden units
    hiddenBiasGradient = compute_gradient(x_image,targets,nnet,hiddenLayer,"bias",[0,76],epsilon)
    check_error(hiddenBiasGradient,-hiddenLayer.d_b[0,76],epsilon**2,"Hidden Bias Gradients")
    hiddenWeight1Gradient = compute_gradient(x_image,targets,nnet,hiddenLayer,"weights",[600,83],epsilon)
    check_error(hiddenWeight1Gradient,-hiddenLayer.d_w[600,83],epsilon**2,"Hidden Weight 1 Gradients")
    hiddenWeight2Gradient = compute_gradient(x_image,targets,nnet,hiddenLayer,"weights",[90,3],epsilon)
    check_error(hiddenWeight2Gradient,-hiddenLayer.d_w[90,3],epsilon**2,"Hidden Weight 2 Gradients")




def compute_gradient(x_image,targets,network,layer,flag,index,epsilon):
    obj = layer.b if flag == "bias" else layer.w
    obj[index[0],index[1]] += epsilon
    upper_cost,logits = network.forward_pass(x_image,targets)
    print("\nupper cost=",upper_cost)
    obj[index[0],index[1]] -= 2*epsilon
    lower_cost,logits = network.forward_pass(x_image,targets)
    print("lower_cost",lower_cost)
    gradient = (upper_cost - lower_cost)/(2*epsilon)
    obj[index[0],index[1]] += epsilon #Restore network!
    print("Numerical approximation gradient=",gradient)
    return gradient




def check_error(n1, n2,margin,print_statement):
    print("Backprop gradient=",n2)
    if np.abs(n1-n2) < margin:
        print(print_statement+" agree!")
    else:
        print(print_statement+" don't agree!")


if __name__=="__main__":
    main()

