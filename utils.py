import matplotlib.pyplot as plt


def out_plots(gt, pred):
    gt = gt.detach().numpy()
    pred = pred.detach().numpy()
    
    fig, ax = plt.subplots(nrows=1, ncols=2)

    plt.subplot(1,2,1)
    for i, trajectory in enumerate(gt):
        x = trajectory[:, 0]  # Extract x positions
        y = trajectory[:, 1]  # Extract y positions
        plt.title("ground_truth_trajectories")
        plt.legend()
        plt.grid(True)
        plt.plot(x, y)

    plt.subplot(1,2,2)
    for i, trajectory in enumerate(pred):
        x = trajectory[:, 0]  # Extract x positions
        y = trajectory[:, 1]  # Extract y positions
        plt.title("Predicted_trajectories")
        plt.legend()
        plt.grid(True)
        plt.plot(x, y)
    
    plt.show()