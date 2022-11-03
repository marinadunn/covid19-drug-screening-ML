import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation


def horizontal_accuracy_chart():
    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(15, 10))

    df = pd.read_csv("Test_Results.csv")
    df_accuracy = df[df["Metric"] == "Accuracy"]

    df_accuracy = df_accuracy.sort_values("Epoch50", ascending=False)[:20]
    print(df_accuracy.head())

    sns.set_color_codes("muted")
    sns.barplot(x="Epoch50", y="Model", data= df_accuracy, label="Model", orient = "h")

    plt.xlabel("Accuracy After 50 Epochs")
    plt.tight_layout()

    plt.savefig("top20_epoch50_hbar_plot.png")


def animated_training():
    df = pd.read_csv("Test_Results.csv")
    df_train = pd.read_csv("Train_Results.csv")

    df_accuracy = df[df["Metric"] == "Accuracy"]

    # Get top 5 models based on train data accuracy
    df_top20_accuracy = df_accuracy.sort_values("Epoch50", ascending=False)[:5]

    # Only get train data from top 5 models
    df_train = df_train[df_train['Model'].isin(df_top20_accuracy['Model'].values.tolist())]

    # Only get Total Loss values
    df_train = df_train[df_train['Metric'] == "Total_Loss"]

    fig = plt.figure()
    ax = plt.axes()

    def animate_func(num):
        ax.clear()  # Clears the figure
                    
        for index, row in df_train.iterrows():
            row_list = row.values.tolist()
            ax.plot(row_list[2:num+2], label=row_list[0])

        # # Setting Axes Limits
        ax.set_xlim([0, 50])
        ax.set_ylim([0, 20])

        # Adding Figure Labels
        ax.set_title('Training Loss at Epoch ' + str())
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        plt.legend()

    line_ani = animation.FuncAnimation(fig, animate_func, interval=50, frames=50) 
    writergif = animation.PillowWriter(fps=50/6)
    line_ani.save('train_animation.gif', writer=writergif)

animated_training()