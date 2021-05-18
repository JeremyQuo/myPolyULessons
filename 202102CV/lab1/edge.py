from PIL import Image # pillow package
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image, ImageDraw

def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr

def save_array_as_img(arr, file):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)

def show_array_as_img(arr):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.show()

def rgb2gray(arr):
    R = arr[:, :, 0] # red channel
    G = arr[:, :, 1] # green channel
    B = arr[:, :, 2] # blue channel
    gray = 0.2989*R + 0.5870*G + 0.1140*B
    return gray

#########################################
## Please complete following functions ##
#########################################

def sobel(arr):
    '''Apply sobel operator on arr and return the result.'''
    # TODO: Please complete this function.
    # your code here
    x_matrix = np.array([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])
    y_matrix = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
    Gx = ndimage.convolve(arr, x_matrix)
    Gy = ndimage.convolve(arr, y_matrix)
    G = np.sqrt(np.square(Gx)+np.square(Gy))

    return G, Gx, Gy

def nonmax_suppress(G, Gx, Gy):
    '''Suppress non-max value along direction perpendicular to the edge.'''
    assert G.shape == Gx.shape
    assert G.shape == Gy.shape
    # TODO: Please complete this function.
    # your code here

    # deal with matrix to decrease running time
    Gx_gradient = np.divide(Gx, G)
    Gy_gradient = np.divide(Gy, G)

    H, W = G.shape

    G_nonzero =np.ones([H,W]).nonzero()

    G_y_index=G_nonzero[0]
    G_x_index=G_nonzero[1]

    max_x_indexList=G_x_index + Gx_gradient.reshape([H*W])
    max_y_indexList=G_y_index+Gy_gradient.reshape([H*W])

    min_x_indexList=G_x_index-Gx_gradient.reshape([H*W])
    min_y_indexList = G_y_index - Gy_gradient.reshape([H*W])

    min_value = ndimage.map_coordinates(G, [min_y_indexList, min_x_indexList]).reshape([H,W])
    max_value = ndimage.map_coordinates(G, [max_y_indexList, max_x_indexList]).reshape([H,W])

    suppressed_G=np.array(G)
    suppressed_G[suppressed_G<min_value]=0

    suppressed_G[suppressed_G<max_value]=0

    return suppressed_G


def nonmax_suppress_loop(G, Gx, Gy):
    '''Suppress non-max value along direction perpendicular to the edge.'''
    assert G.shape == Gx.shape
    assert G.shape == Gy.shape
    # TODO: Please complete this function.
    # your code here

    # original loop
    Gx_gradient = np.divide(Gx, G)
    Gy_gradient = np.divide(Gy, G)
    min_indexList = [[], []]
    max_indexList = [[], []]

    for index, x in np.ndenumerate(G):
        min_indexList[0].append(index[0] - Gy_gradient[index[0]][index[1]])
        min_indexList[1].append(index[1] - Gx_gradient[index[0]][index[1]])
        max_indexList[0].append(index[0] + Gy_gradient[index[0]][index[1]])
        max_indexList[1].append(index[1] + Gx_gradient[index[0]][index[1]])

    min_indexList = np.reshape(ndimage.map_coordinates(G, min_indexList),G.shape)
    max_indexList = np.reshape(ndimage.map_coordinates(G, max_indexList),G.shape)
    suppressed_G = np.zeros_like(G)
    for i in range(len(G)):
        for j in range(len(G[i])):
            if(G[i][j]> min_indexList[i][j] and  G[i][j]> max_indexList[i][j]):
                suppressed_G[i][j]=G[i][j]
    show_array_as_img(suppressed_G)
    return suppressed_G

def thresholding(G, t):
    '''Binarize G according threshold t'''
    # TODO: Please complete this function.
    # your code here
    # threshold

    # binarize G
    G[G <= t] = 0
    G[G > t] = 255

    return G
min=0
pi_list=None
def hough(G):
    '''Return Hough transform of G'''
    # TODO: Please complete this function.
    # your code here
    # create array pi and calculate cos and sin
    global min
    global pi_list
    pi_list=np.linspace(0.0, np.pi, 500, endpoint=True)
    cos_list=np.cos(pi_list)
    sin_list=np.sin(pi_list)

    # extract all the edge point
    temp=np.where(G>0)

    # calculate the p and show the result
    r = sin_list.reshape(-1, 1) * temp[0] + cos_list.reshape(-1, 1) * temp[1]
    r = r.astype(int)
    min = np.min(r)
    max=np.max(r)
    result_list=np.zeros([max-min+1,len(pi_list)], dtype=int)
    # build the accumulator array
    for i in range(len(r)):
        row = r[i]
        temp=Counter(row)
        for key in temp:
            index=key-min
            result_list[index][i]=temp[key]

    plt.title("Hough Transform")
    sns.heatmap(result_list)
    # plt.plot(pi_list, r, color='black', linewidth=0.003)
    plt.gca().invert_yaxis()
    # hide the x and y axis
    plt.xticks([])
    plt.yticks([])
    plt.savefig('hough.jpg')
    plt.show()
    return result_list

def local_maxima(result_list):
    for i in range(1,len(result_list)-1):
        for j in range(1,len(result_list[i])-1):
            increment_x = [-1,0,1]
            increment_y = [-1,0,1]
            for x in increment_x:
                for y in increment_y:
                    if result_list[i+y][j+x]>result_list[i][j]:
                        result_list[i][j]=0
    # result.set_index(['theta'], inplace=True)
    return result_list



def draw_line(result_list,boundary):
    # boundary means the number of points on a same line
    # according to theta and p to choose 2 points to draw a line
    global min
    global pi_list
    line_list=np.where(result_list>boundary)
    temp=np.zeros_like(line_list,dtype=float)

    for i in range(len(line_list[0])):
        temp[0][i]=line_list[0][i]+min
    for i in range(len(line_list[1])):
        temp[1][i]=pi_list[line_list[1][i]]
    im = Image.open("road.jpeg")
    draw = ImageDraw.Draw(im)
    for i in range(len(temp[0])):
        y1=temp[0][i]/np.sin(temp[1][i])
        y2=(temp[0][i]-im.size[0]*np.cos(temp[1][i]))/np.sin(temp[1][i])
        draw.line((0,y1)+(im.size[0],y2), fill=3)
    im.save('detection_result.jpg', quality=95)
    im.show()

def guassian(gray):
    gauss = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]])
    gauss = gauss / gauss.sum()  # normalize the sum to be 1
    out = ndimage.convolve(gray, gauss)
    return out

def main():

    img = read_img_as_array('road.jpeg')
    arr=rgb2gray(img)
    save_array_as_img(arr,'gray.jpg')
    guassian_out=guassian(arr)
    # guassian_out = ndimage.gaussian_filter(arr, sigma=2)
    save_array_as_img(guassian_out, 'gauss.jpg')
    G, Gx, Gy=sobel(guassian_out)
    save_array_as_img(G, 'G.jpg')
    save_array_as_img(Gx, 'Gx.jpg')
    save_array_as_img(Gy, 'Gy.jpg')

    G=nonmax_suppress(G, Gx, Gy)
    save_array_as_img(G, 'supress.jpg')

    G=thresholding(G,200)
    save_array_as_img(G, 'edgemap.jpg')
    draw_line(local_maxima(hough(G)),boundary=86)
    #TODO: detect edges on 'img'

main()