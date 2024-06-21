"""
@Project  : t-conan
@File     : plot_pie.py
@Author   : Shaobo Cui
@Date     : 24.02.23 10:49
"""

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

outpie_radius = 1.2
innerpie_radius = 0.8

def plot_pie(df, ax, time_intervals):
    inner_labels = time_intervals
    data_df = df[time_intervals]

    # Plot outer pie.
    outer_pie_values = []
    for column in data_df:
        for value in data_df[column]:
            outer_pie_values.append(value)

    outer_colors = cmap(np.arange(0, 10, 1))
    out_colors = []
    for j in range(len(time_intervals)):
        for color in outer_colors:
            out_colors.append(color)

    ax.pie(outer_pie_values,
            startangle=90, pctdistance=0.88, colors=outer_colors,
            radius=outpie_radius, labeldistance=1.05,
            textprops={'fontweight': 'bold', 'fontsize': 13},
            wedgeprops={'linewidth': 0.3, 'edgecolor': "w"})

    # Plot inner pie.
    inner_colors = cmap(12 + np.array(range(len(time_intervals))))

    # inner_pie_values = df[['Sum']][0:len(time_intervals)].values.tolist()
    # inner_pie_values = [a[0] for a in inner_pie_values]

    inner_pie_values = np.add.reduceat(outer_pie_values, np.arange(0, len(outer_pie_values), 10))

    # PLotting the inner pie
    ax.pie(inner_pie_values, labels=inner_labels,
                 startangle=90, pctdistance=0.05, colors=inner_colors,
                 radius=innerpie_radius, labeldistance=0.5, rotatelabels =True,
                 textprops={'weight': 'bold', 'fontsize': 15},
                 wedgeprops={'linewidth': 0.3, 'edgecolor': "w"})



    # Creating the donut shape for the pie
    centre_circle = plt.Circle((0, 0), 0.25, fc='white')
    ax.add_patch(centre_circle)  # adding the centre circle

    return ax

if __name__ == "__main__":
    cmap = plt.get_cmap("tab20")

    df = pd.read_csv('stats.csv', sep='&')
    x = ['Environment', 'Business', 'Sci-Tech', 'Health', 'Work', 'Politics',
         'Education', 'Sports', 'Entertainment', 'Travel']

    short_term_intervals = ['instantly', 'seconds', 'minutes', 'hours', 'days', 'weeks']
    long_term_intervals = ['months', 'years', 'decades', 'centuries']

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax_short = plot_pie(df, ax1, short_term_intervals)
    ax_long = plot_pie(df, ax2, long_term_intervals)
    fig.legend(x, loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=12, fancybox=True, shadow=True, prop={'size': 15})
    fig.set_figheight(10)
    fig.set_figwidth(25)
    fig.savefig('pie_plot.pdf')
    """

    # Taking raw data of three students
    source_data={'students':['Jake','Amy','Boyle'],
    'math_score':[68,82,97],
    'english_score':[70,93,99],
    'physics_score':[73,85,95]}

    # Segregating the raw data into usuable form
    df=pd.DataFrame(source_data,columns=
    ['students','math_score','english_score','physics_score'])
    df['cumulative_score']=df['math_score']+df['english_score']
    +df['physics_score']

    # Seperating the sub-parts of the given data
    x1= df.iloc[0:3,1]
    x2= df.iloc[0:3,2]

    # Setting figure colors
    outer_colors = cmap(np.arange(3)*4)
    inner_colors = cmap(np.array([1,5,9]))

    # Setting the size of the figure
    plt.figure(figsize=(8,6))

    # Plotting the outer pie
    x1 = [1, 2, 2, 3, 4, 5, 5, 7, 9]
    labels = ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']
    outer_colors = cmap(np.arange(3)*3)
    out_colors = []
    for j in range(3):
        for color in outer_colors:
            out_colors.append(color)

    plt.pie(x1,
           startangle=90, pctdistance =0.88 ,colors=outer_colors,
           radius= 1.0, labeldistance=1.05,
           textprops ={ 'fontweight': 'bold','fontsize':13},
           wedgeprops = {'linewidth' : 3, 'edgecolor' : "w" } )

    # PLotting the inner pie
    plt.pie(x2,startangle=90, pctdistance =0.85,colors=inner_colors,
            autopct = '%1.1f%%',radius= 0.60,
           textprops ={'fontweight': 'bold' ,'fontsize':13},
           wedgeprops = {'linewidth' : 3, 'edgecolor' : "w" } )

    # Creating the donut shape for the pie
    centre_circle = plt.Circle((0,0), 0.25, fc='white')
    fig= plt.gcf()
    fig.gca().add_artist(centre_circle) # adding the centre circle

    # Plotting the pie
    plt.axis('equal')  # equal aspect ratio
    plt.legend(['a', 'b', 'c'], loc=4, fontsize =15)
    plt.tight_layout()
    plt.show()
    """
