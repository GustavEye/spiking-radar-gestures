import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 

def get_false(i, out_label_all, true_label_all, true_label=None, false_label=None):
    false_classification = np.argwhere(out_label_all != true_label_all)
    
    if true_label is None and false_label is None:
        return false_classification[i][0]
    
    elif true_label is not None:
        true_label_wrong = true_label_all[false_classification]
    
        i_number = np.argwhere(true_label_wrong == true_label)
        return false_classification[i_number[i][0]][0]
    
    else:
        out_label_wrong = out_label_all[false_classification]
    
        i_number = np.argwhere(out_label_wrong == false_label)
        return false_classification[i_number[i][0]][0]

def get_true(i, out_label_all, true_label_all, true_label=None):
    true_classification = np.argwhere(out_label_all == true_label_all)

    if true_label is None:
        return true_classification[i][0]
    else:
        label_correct = true_label_all[true_classification]
    
        i_number = np.argwhere(label_correct == true_label)
        return true_classification[i_number[i][0]][0]

def plot_voltage(ax, v, z=0):
    ax.plot(v)
    ax.hlines(0.1, xmin=0, xmax=len(v), linestyles='dashed')
    ax.vlines(np.argwhere(z == 1.0), ymin=np.min(v), ymax=np.max(v)+0.01, linestyles='dashed')

def spike_train_classes_plot(input_z, mid_z, mid_v, out_z, true_label=None, figsize=(10,8), font_size=14.0):
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10,8))

    ax[0].spy(input_z.transpose(), aspect='auto')
    ax[0].set_title('(a)', loc='right', fontsize = font_size)
    ax[0].set_ylabel('Neuron index\n(input)', fontsize = font_size)
    ax[1].spy(mid_z.transpose(), aspect='auto')
    ax[1].set_title('(b)', loc='right', fontsize = font_size)
    ax[1].set_ylabel('Neuron index\n(hidden)', fontsize = font_size)

    out_shape = np.shape(out_z)
    for i in range(out_shape[1]):
        if i == true_label:
            lw = 2
            c = 'g'
        else:
            lw = 1
            c = '0.5'
        ax[2].plot(out_z[:,i], linewidth=lw, color=c)
    ax[2].set_title('(c)', loc='right', fontsize = font_size)
    ax[2].set_ylabel('Class\nprobability', fontsize = font_size)

    if true_label is not None:
        fig.suptitle('true label: ' + str(true_label) + ', ' + str(true_label == np.argmax(out_z[-1,:])))

    plot_num = 20
    plot_voltage(ax[3], mid_v[:,plot_num], mid_z[:,plot_num])
    ax[3].set_title('(d)', loc='right', fontsize = font_size)
    ax[3].set_ylabel('Membrane\nvoltage', fontsize = font_size)
    ax[3].set_xlabel('Timestep', fontsize = font_size)

    return fig

def spike_train_classes_plot_2(input_z, mid_z, mid_z_2, out_z, true_label=None, figsize=(10,8), font_size=14.0):
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10,8))

    ax[0].spy(input_z.transpose(), aspect='auto')
    ax[0].set_title('(a)', loc='right', fontsize = font_size)
    ax[0].set_ylabel('Neuron index\n(input max)', fontsize = font_size)
    ax[1].spy(mid_z.transpose(), aspect='auto')
    ax[1].set_title('(b)', loc='right', fontsize = font_size)
    ax[1].set_ylabel('Neuron index\n(input mean)', fontsize = font_size)
    ax[2].spy(mid_z_2.transpose(), aspect='auto')
    ax[2].set_title('(c)', loc='right', fontsize = font_size)
    ax[2].set_ylabel('Neuron index\n(hidden)', fontsize = font_size)

    out_shape = np.shape(out_z)
    for i in range(out_shape[1]):
        if i == true_label:
            lw = 2
            c = 'g'
        else:
            lw = 1
            c = '0.5'
        ax[3].plot(out_z[:,i], linewidth=lw, color=c)
    ax[3].set_title('(d)', loc='right', fontsize = font_size)
    ax[3].set_ylabel('Class\nprobability', fontsize = font_size)

    #if true_label is not None:
    #    fig.suptitle('true label: ' + str(true_label) + ', ' + str(true_label == np.argmax(out_z[-1,:])))

    return fig

def confusion_matrix_plot(true_label_all, out_label_all, figsize=(6,6)):
    cm = confusion_matrix(true_label_all, out_label_all)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    disp.plot(ax=ax, colorbar=False)

    return fig

def component_plot(result, label, enable3D=True, title_label='Component Analysis', figsize=(15,10), scattersize=1, enable_legend=True):
    """
    Function to plot T-SNE or PCA results in 2D or 3D depending on the dimension of the result matrix.
    Right now there is no support for subplots of a 3D plot. 
    :data result : the results of the fit_tranform function for PCA and T-SNE
    :data label  : label vector with same samples as results
    :param title_label : title text of the plot
    """
    fig = plt.figure(figsize=figsize)
        
    # Extract dimension and number of classes
    dimension = np.shape(result)[1]
    nr_classes = len(np.unique(label))

    # setup color coding
    cmap = ListedColormap(sns.color_palette("hls", n_colors=nr_classes).as_hex())
    color=np.array(label).astype(int)
    
    # Plot Function for 2D and 3D
    if dimension >= 2 and enable3D == True:
        # 3D plot 
        ax = fig.add_subplot(111, projection='3d')

        # 3D scatter plot
        sc = ax.scatter(result[:,0], result[:,1], result[:,2], s=scattersize, c=color.tolist(), marker='o', cmap=cmap, alpha=0.75)
        ax.set(title=title_label, xlabel='x axis', ylabel='y axis', zlabel='z axis')
        if enable_legend:
            legend = ax.legend(*sc.legend_elements(),title="Classes")
    else:
        # 2D plot
        ax = fig.add_subplot(111)

        # 3D scatter plot
        sc = ax.scatter(result[:,0], result[:,1], s=scattersize, marker='o', c=color, cmap=cmap, alpha=0.75)
        ax.set(title=title_label, xlabel='x axis', ylabel='y axis')
        if enable_legend:
            legend = ax.legend(*sc.legend_elements(),title="Classes")
    
    return fig

def component_plot_ax(ax, result, label, title_label='Component Analysis', scattersize=1, enable_legend=False, enable_ticks=False):
    """
    Function to plot T-SNE or PCA results in 2D or 3D depending on the dimension of the result matrix.
    Right now there is no support for subplots of a 3D plot. 
    :data result : the results of the fit_tranform function for PCA and T-SNE
    :data label  : label vector with same samples as results
    :param title_label : title text of the plot
    """        
    
    ax.tick_params(left=enable_ticks,
                bottom=enable_ticks,
                labelleft=enable_ticks,
                labelbottom=enable_ticks)
    
    nr_classes = len(np.unique(label))

    # setup color coding
    cmap = ListedColormap(sns.color_palette("hls", n_colors=nr_classes).as_hex())
    color=np.array(label).astype(int)
    
    sc = ax.scatter(result[:,0], result[:,1], s=scattersize, marker='o', c=color, cmap=cmap, alpha=0.75)
    #ax.set(title=title_label, xlabel='x axis', ylabel='y axis')
    ax.set(title=title_label)
    if enable_legend:
        legend = ax.legend(*sc.legend_elements(),title="Classes")