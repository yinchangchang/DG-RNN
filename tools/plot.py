# coding=utf8
import matplotlib.pyplot as plt
import numpy as np

def plot_multi_graph(image_list, name_list, save_path=None, show=False):
    graph_place = int(np.sqrt(len(name_list) - 1)) + 1
    print 'eeee'
    for i, (image, name) in enumerate(zip(image_list, name_list)):
        ax1 = plt.subplot(graph_place,graph_place,i+1)
        ax1.set_title(name)
        # plt.imshow(image,cmap='gray')
        plt.imshow(image)
        plt.axis('off')
        print 'eeee'
    if save_path:
        plt.savefig(save_path)
        print 'eeee'
        pass
    if show:
        plt.show()

def plot_multi_line(x_list, y_list, name_list, save_path=None, show=False):
    plt.clf()
    graph_place = int(np.sqrt(len(name_list) - 1)) + 1
    for i, (x, y, name) in enumerate(zip(x_list, y_list, name_list)):
        ax1 = plt.subplot(graph_place,graph_place,i+1)
        # ax1 = plt.subplot(1,graph_place,i+1)
        fontsize= 15
        ax1.set_title(name, fontsize=fontsize)
        plt.plot(x,y, markersize=12)
        
        plt.xticks([-120, -90, -60, -30, -7])
        plt.xlim(-130, 0)
        # plt.imshow(image,cmap='gray')
        plt.xlabel('Hold-Off Window', fontsize=fontsize)
        if i == 0:
            plt.ylabel('Comtribution Rate', fontsize=fontsize)
            plt.ylim(0.855, 0.915)
        else:
            plt.ylim(0.085, 0.145)
    if save_path:
        plt.savefig(save_path+'.eps')
    if show:
        plt.show()


def plot_multi_line_one_image(x_list, y_list, name, save_path=None, show=False):
    plt.clf()
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        ax1 = plt.subplot(1,1,1)
        plt.plot(x,y)
        if i == 0:
            ax1.set_title(name)
            plt.xlabel('Hold-off window')
            plt.ylabel('Comtribution rate')
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
