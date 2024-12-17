import matplotlib.pyplot as plt

def interval_set_comparison(set_one,set_two,set_one_id,set_two_id,title):
    """
        Function that plots the different intervals from the two sets next to each other by their indices. It is assumed that the two sets
        contain intervals where each index from both sets represents the same underlying value. The y-Axis is a number line representing the 
        resepective values of the intervals. The x-Axis represents the index set over the two sets
    """

    assert(len(set_one)==len(set_two)) # Core assertion if this is not given something is fundamentally wrong
    num_intervals = len(set_one)
    fig,axis = plt.subplots()
    for i, (int1,int2) in enumerate(zip(set_one,set_two)):
        axis.plot([i - 0.1, i - 0.1], [int1.lower, int1.upper], color='blue', label=set_one_id if i == 0 else "")
        axis.plot([i + 0.1, i + 0.1], [int2.lower, int2.upper], color='red', label=set_two_id if i == 0 else "")
        axis.scatter([i - 0.1, i - 0.1], [int1.lower, int1.upper], color='blue')
        axis.scatter([i + 0.1, i + 0.1], [int2.lower, int2.upper], color='red')
    
    axis.set_xticks(range(num_intervals))
    axis.set_xticklabels([f"Node {i}" for i in range(num_intervals)])
    axis.set_xlabel("Index")
    axis.set_ylabel("Value Range")
    axis.set_title(title)
    axis.legend()
    axis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def scatter_montecarlo(vecs):
    if len(vecs[0]) == 2:
        x_cords = [vec[0] for vec in vecs]
        y_cords = [vec[1] for vec in vecs]
        fig,axs = plt.subplots()
        axs.scatter(x_cords,y_cords)
        plt.show()
    else:
        print("Not yet implemented but t-SNE reduction code here")