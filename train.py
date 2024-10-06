import sys
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.example_libraries.optimizers import sgd
from jax.nn import log_softmax
from model import init_model

with jnp.load("mnist.npz") as data:
    x_train = data["x_train"].reshape(60000, 784) # flatten out
    y_train = data["y_train"]
    x_test  = data["x_test"].reshape(10000, 784)
    y_test  = data["y_test"]

def eval_loss(model, img, truth):
    logits = model(img)
    preds = log_softmax(logits)
    one_hot_y = jnp.where(jnp.arange(10) != truth, 0, 1) # put y into one hot encoding
    return -1 * (one_hot_y * preds).mean()

def train(epochs=3, lr=.001):
    model = init_model(jax.random.PRNGKey(0), 784, 256, 10)
    opt_init, opt_update, get_params = sgd(lr)
    opt_state = opt_init(model)
    evalfn = jax.jit(jax.value_and_grad(eval_loss))

    losses = []
    train_len = x_train.shape[0]
    for epoch in range(epochs):
        print(f"== on epoch {epoch}")
        # training loop 
        for i, (img, label) in enumerate(zip(x_train, y_train)):
            loss, grads = evalfn(get_params(opt_state), img, label)
            opt_state = opt_update(i*(epoch+1), grads, opt_state)
            if i % 1000 == 0:
                sys.stdout.write(
                    f"{i}/{train_len} ({i/train_len*100:.2f}%) "
                    f"loss: {loss.item()}\r")
                losses.append(loss)

        # testing/evalution using test dataset
        num_correct = 0
        for i, (img, label) in enumerate(zip(x_test, y_test)):
            if get_params(opt_state)(img).argmax() == label:
                num_correct += 1
        print(
            "\n"
            f"acc: ({num_correct/x_test.shape[0]*100:.2f}%) | "
            f"correct: {num_correct} | "
            f"wrong: {x_test.shape[0] - num_correct}")

    xt = x_test[0]
    yt = y_test[0]

    # demonstrate a prediction
    plt.figure()
    plt.title(f"real: {yt}, prediction: {get_params(opt_state)(xt).argmax()}")
    plt.imshow(xt.reshape(28, 28))
    
    # plot how loss value changed over time
    plt.figure()
    plt.title("model performance")
    plt.plot(losses, label="loss value")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    return get_params(opt_state) # the optimized model

train()