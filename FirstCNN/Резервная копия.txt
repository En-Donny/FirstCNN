import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy
import numpy as np
import scipy.special  # библиотека scipy.special содержит сигмоиду expit()
from keras.datasets import mnist


def proverka(ker, inp, prev):
    """
    Эта функция - что-то вроде юнит-теста для функции обратной свертки в случае вычисления ошибки входа
    """
    kernel_change = ker
    delta_inp = np.zeros(inp.shape)
    raznost = prev.shape[0]-1  # расчет разности размерностей входных слоев и фильтров свертки
    print(kernel_change)
    kernel_change = np.rot90(kernel_change, 2)
    print('\n', kernel_change)
    kernel_change = np.flip(kernel_change, 1)
    # if raznost % 2 != 0:
    #     kernel_change = np.pad(kernel_change,
    #                            [(int(raznost / 2), int(raznost / 2 + 1)), (int(raznost / 2), int(raznost / 2 + 1))],
    #                            mode='constant', constant_values=0)
    # else:
    kernel_change = np.pad(kernel_change,
                           [(int(raznost), int(raznost)), (int(raznost), int(raznost))],
                           mode='constant', constant_values=0)
    i = kernel_change.shape[0]-1
    y_index_out = 0  # индекс массива результата по строкам
    while (i - prev.shape[0]) >= 0:
        x_index_out = 0  # индекс массива результата по строкам
        j = kernel_change.shape[1] - 1
        while(j - prev.shape[1]) >= 0:
            virez = (kernel_change[i - prev.shape[0]+1:i+1, j - prev.shape[1]+1:j+1])
            delta_inp[y_index_out][x_index_out] = np.sum(virez * prev)
            x_index_out += 1
            j -= 1

        y_index_out += 1
        i -= 1
    print(delta_inp.shape)


def derivative_sigm_matrix(mat):
    return scipy.special.expit(mat)*(1 - scipy.special.expit(mat))  # производная по функции сигмойды задается
    # через sigm(1 - sigm)


def flatten(input):
    output = np.array([])
    for i in range(input.shape[0]):
        output = np.append(output, [np.ravel(input[i])])
    return(output)


def init_net(input_nods_count):

    input_nodes = int(input_nods_count)  # входной слой (28*28 изображения подаются, поэтому нейронов на входном слое
    # 784 под каждое значение
    #пикселя )

    hidden_nodes = int(input('Введите число скрытых нейронов: ')) #скрытый слой
    out_nodes = 10 #выходной слой, как раз из 10, так как у нас всего 10 цифр

    learn_node = float(input('Введите скорость обучения: '))
    return input_nodes, hidden_nodes, out_nodes, learn_node


def creat_net(input_nodes, hidden_nodes, out_nodes):
    # сознание массивов. -0.5 вычитаем что бы получить диапазон -0.5 +0.5 для весов
    # веса аксонов между входным и скрытым слоем
    input_hidden_w = (numpy.random.rand(hidden_nodes, input_nodes) - 0.5)
    # веса аксонов между скрытым и выходным слоем
    hidden_out_w = (numpy.random.rand(out_nodes, hidden_nodes) - 0.5)
    return input_hidden_w, hidden_out_w


def fun_active_matrix(input):
    out = np.zeros((input.shape))
    for num in range(out.shape[0]):
        for i in range(out.shape[1]):
            for j in range(out.shape[2]):
                out[num][i][j] = scipy.special.expit(input[num][i][j])
    return out


def fun_active_vector(x):
    return scipy.special.expit(x)


def relu(matrix):
    out_matrix = np.zeros(matrix.shape)
    for num in range(matrix.shape[0]):
        for i in range(matrix.shape[1]):
            for j in range(matrix.shape[2]):
                out_matrix[num][i][j] = max(0.0, matrix[num][i][j])
    return out_matrix


def derevative_relu(matrix):
    out_matrix = np.zeros(matrix.shape)
    for num in range(matrix.shape[0]):
        for i in range(matrix.shape[1]):
            for j in range(matrix.shape[2]):
                if matrix[num][i][j] >= 0:
                    out_matrix[num][i][j] = 1
                else:
                    out_matrix[num][i][j] = 0
    return out_matrix

def convolution(inp, kernel, stride):
    # создали массив заполненный пустотой и размерностью будещего массива
    if(inp.shape[1] < kernel.shape[1]):
        return inp
    else:
        out_matrix = np.empty((inp.shape[0]*kernel.shape[0], int((inp.shape[1] - kernel.shape[1])/stride + 1),
                               int((inp.shape[2] - kernel.shape[2])/stride + 1)))
        num_out_map = 0
        for input_map in range(inp.shape[0]):
            for num_kernel in range(kernel.shape[0]):
                i = 0
                y_index_out = 0  # индекс массива результата по строкам
                while (i+kernel.shape[1]) <= inp.shape[1]:
                    x_index_out = 0  # индекс массива результата по столбцам
                    j = 0
                    while(j+kernel.shape[1]) <= inp.shape[2]:
                        virez = (inp[input_map][i:i+kernel.shape[1], j:j+kernel.shape[1]])  # сохраняем в отдельный
                        # массив рецептивное поле
                        out_matrix[num_out_map][y_index_out][x_index_out] = np.sum(virez * kernel[num_kernel])  # каждый
                        # элемент новой матрицы равен
                        # почленовому произведению матрицы фильтра и рецептивного поля матрицы, на которое накладывается
                        # фильтр
                        x_index_out += 1
                        j += stride
                    y_index_out += 1
                    i += stride
                num_out_map += 1
        return out_matrix


def maxpool(inp, kernel_size, stride):
    if(inp.shape[1] % 2 != 0):  # если матрица со сторонами не четной длины (то есть, не получится целое количесвто
        # раз пройти фильтром по какой-то из размерностей ) ==> добавляем паддинги к матрице справа или снизу
        # (или и там, и там)
        new = np.zeros((inp.shape[0], inp.shape[1], 1))
        inp = np.concatenate((inp, new), axis=2)
        new = np.zeros((inp.shape[0], 1, inp.shape[2]))
        inp = np.concatenate((inp, new), axis=1)
    mask = np.zeros(inp.shape)  # инициализируем маску, понадобится при бэкварде
    down_sampled_matrix = np.empty((inp.shape[0], int((inp.shape[1] - kernel_size)/stride + 1),
                                    int((inp.shape[2] - kernel_size)/stride + 1)))
    for num in range(inp.shape[0]):
        i = 0
        y_index_out = 0  # индекс массива результата по строкам
        while (i + kernel_size) <= inp.shape[1]:
            x_index_out = 0  # индекс массива результата по столбцам
            j = 0
            while (j + kernel_size) <= inp.shape[2]:
                virez = (inp[num][i:i + kernel_size, j:j + kernel_size])  # все то же самое ==> выделяем рецептивное
                # поле
                down_sampled_matrix[num][y_index_out][x_index_out] = np.amax(virez)  # находим максимум рецептивного
                # поля
                mask[num][i:i + kernel_size, j:j + kernel_size][np.where(virez == np.amax(virez))] = 1  # запоминаем
                # место в маске
                x_index_out += 1
                j += stride
            y_index_out += 1
            i += stride

    return inp, down_sampled_matrix, mask


def conv_backward(d_prev, input, kernel, stride):
    kernel_change = kernel
    delta_filters = np.zeros(kernel.shape)
    delta_inputs = np.zeros(input.shape)
    offset_kernels_num = 0  # для каждого входа перед сверткой есть свой набор выходов, но количесвто элементов в
    #  в таких наборах одинаковое, поэтому разделим их вот такой переменной и будем идти в цикле по размеру кажого
    #  набора

    # сначала высчитаем ошибки по фильтрам на данном слое свертки
    for num_input in range(input.shape[0]):
        for num_kernel in range(kernel.shape[0]):
            i = 0
            y_index_out = 0  # индекс массива результата по строкам
            while (i + d_prev.shape[1]) <= input.shape[1]:
                x_index_out = 0  # индекс массива результата по столбцам
                j = 0
                while (j + d_prev.shape[2]) <= input.shape[2]:
                    virez = (input[num_input][i:i + d_prev.shape[1], j:j + d_prev.shape[2]])
                    # как и в прямой свертке, здесь мы делаем свертку выхода и рецептивного поля входа
                    # реализация взята с визуального представления на сайте
                    # https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
                    delta_filters[num_kernel][y_index_out][x_index_out] += np.sum(virez * d_prev[num_kernel +
                                                                                                 offset_kernels_num])
                    x_index_out += 1
                    j += stride

                y_index_out += 1
                i += stride

        offset_kernels_num += kernel.shape[0]  # добавляем смещение, чтобы перейти к следующему набору выходов

        # теперь перейдем к расчету ошибок для входных изображений
    raznost = d_prev.shape[1]-1  # расчет разности размерностей входных слоев и фильтров свертки
    kernel_change = np.flip(np.flip(kernel_change, 2), 1)  # нужно развернуть матрицу на 180' по часовой стрелке (
    # представление взято с сайта https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c)
    kernel_change = np.pad(kernel_change,  # чтобы получить ошибку матрицы входа, мы должны добавить такие паддинги к
                           # повернутому фильтру, чтобы при свертке его с выходом получилась матрица того же размера,
                           # что и входная
                           [(0, 0), (int(raznost), int(raznost)), (int(raznost), int(raznost))],
                           mode='constant', constant_values=0)
    offset_kernels_num = 0

    for num_inputs in range(input.shape[0]):
        for num_kernel in range(kernel_change.shape[0]):
            i = kernel_change.shape[1] - 1
            y_index_out = 0  # индекс массива результата по строкам
            while (i - d_prev.shape[1]) >= 0:
                x_index_out = 0  # индекс массива результата по строкам
                j = kernel_change.shape[2] - 1
                while (j - d_prev.shape[2]) >= 0:
                    virez = (kernel_change[num_kernel][i - d_prev.shape[1] + 1:i + 1, j - d_prev.shape[2] + 1:j + 1])
                    delta_inputs[num_inputs][y_index_out][x_index_out] += np.sum(virez * d_prev[num_kernel+ offset_kernels_num])
                    x_index_out += 1
                    j -= 1
                i -= 1
                y_index_out += 1
        offset_kernels_num += kernel.shape[0]

    return delta_filters, delta_inputs


def maxpool_backward(mask, delta_maxpool, kernel_size, stride):
    # здесь берем за основу маску полученную при макспулл операции и домножаем ее на ошибку выхода текущего слоя
    out = mask
    for num in range(mask.shape[0]):
        i = 0
        y_index_out = 0  # индекс массива результата по строкам
        while (i + kernel_size) <= mask.shape[1]:
            x_index_out = 0  # индекс массива результата по столбцам
            j = 0
            while (j + kernel_size) <= mask.shape[2]:
                out[num][i:i + kernel_size, j:j + kernel_size] *= delta_maxpool[num][y_index_out][x_index_out]
                x_index_out += 1
                j += stride
            y_index_out += 1
            i += stride

    return out





def train(inp, kernel1, kernel2, target, input_hidden_w, hidden_out_w, learn_node):
    """
    Функция тренировки
    """
    # сворачиваем
    conv1 = convolution(inp, kernel1, 1)
    # макспулим
    relu1 = relu(conv1)
    relu1, maxpool1, mask1 = maxpool(relu1, 2, 2)
    # после операции макспулла могли иизмениться размерности матриц, поэтому изменяем и размерность матрицы входа
    conv1 = np.pad(conv1, [(0, 0), (0, relu1.shape[1] - conv1.shape[1]), (0, relu1.shape[2] - conv1.shape[2])],
                   mode='constant', constant_values=0)
    # сворачиваем
    conv2 = convolution(maxpool1, kernel2, 1)
    relu2 = relu(conv2)
    # макспулим
    relu2, maxpool2, mask2 = maxpool(relu2, 2, 2)

    # после операции субдискредизации размерности матриц могли изменить относительно их начальной разницы в размерах
    # (так как мы делаем паддинг в (1,1)  слой нулей снизу и справа если матрица нечетных размеров)
    # поэтому сделаем паддинг для отсальных матриц
    conv2 = np.pad(conv2, [(0, 0), (0, relu2.shape[1] - conv2.shape[1]), (0, relu2.shape[2] - conv2.shape[2])],
                   mode='constant', constant_values=0)
    maxpool1 = np.pad(maxpool1, [(0, 0), (0, conv2.shape[1] - 1 + kernel2.shape[1] - maxpool1.shape[1]),
                                 (0, conv2.shape[2] - 1 + kernel2.shape[2] - maxpool1.shape[2])],
                      mode='constant', constant_values=0)

    # преобразуем получившиеся матрицы в один большой вектор
    fc_input = flatten(maxpool2)
    # fc_input = (numpy.asfarray(fc_input) / 255.0 * 0.99) + 0.01
    # передаем этот вектор в полносвязную сеть
    targets = numpy.zeros(10) + 0.01

    targets[target] = 0.99
    targets = numpy.array(targets, ndmin=2).T
    inputs_sig = numpy.array(fc_input, ndmin=2).T

    # перемножаем матрицу весов аксонов на вектор входных данных
    hidden_inputs = numpy.dot(input_hidden_w, inputs_sig)
    # считаем выходы скрытого слоя через функию активации
    hidden_out = fun_active_vector(hidden_inputs)
    # умножаем матрицу весов между скрытым и выходным слоем на вектор выходных значений скрытого слоя
    final_inputs = numpy.dot(hidden_out_w, hidden_out)

    # считаем выходы всей нейронной сети
    final_out = fun_active_vector(final_inputs)
    # Рассчитываем ошибку выходного слоя
    out_errors = targets - final_out

    delta_out = out_errors * final_out * (1 - final_out)
    # Рассчитываем ошибку скрытого слоя
    hidden_errors = numpy.dot(hidden_out_w.T, delta_out)
    delta_hidden = hidden_errors * hidden_out * (1 - hidden_out)

    # Обновление весов связей
    # веса между скрытым и выходным слоем
    hidden_out_w += learn_node * numpy.dot(delta_out, numpy.transpose(hidden_out))
    input_hidden_w += learn_node * numpy.dot(delta_hidden, numpy.transpose(inputs_sig))

    input_errors = numpy.dot(input_hidden_w.T, delta_hidden)

    """
    Здесь начинается передача ошибок в свреточные слои сети
    """
    delta_maxpool2 = input_errors.reshape(maxpool2.shape)  # берем дельты ошибок с полносвязного слоя и объединяем
    # их назад в матрицу

    delta_relu2 = maxpool_backward(mask2, delta_maxpool2, 2, 2)  # находим дельту предыдущего выхода по
    # маске этого выхода
    delta_conv2 = delta_relu2 * derevative_relu(relu2)
    delta_kernel2, delta_maxpool1 = conv_backward(delta_conv2, maxpool1, kernel2, 1)  # расчитываем ошибки для фильтров
    # # # и ошибки для входных матриц
    kernel2 += learn_node * delta_kernel2  # обучаем фильтр

    delta_relu1 = maxpool_backward(mask1, delta_maxpool1, 2, 2)
    delta_conv1 = delta_relu1 * derevative_relu(relu1)
    delta_kernel1, delta_maxpool0 = conv_backward(delta_conv1, inp, kernel1, 1)
    kernel1 += learn_node * delta_kernel1
    return kernel1, kernel2, input_hidden_w, hidden_out_w


def forward(inp, kernel1, kernel2, input_hidden_w, hidden_out_w):

    """
    Функция предсказания
    """
    # сворачиваем
    conv1 = convolution(inp, kernel1, 1)

    relu1 = relu(conv1)
    # макспулим
    relu1, maxpool1, mask1 = maxpool(relu1, 2, 2)

    # сворачиваем
    conv2 = convolution(maxpool1, kernel2, 1)
    # макспулим
    relu2 = relu(conv2)
    relu2, maxpool2, mask2 = maxpool(relu2, 2, 2)

    # преобразуем получившиеся матрицы в один большой вектор
    fc_input = flatten(maxpool2)
    # fc_input = (numpy.asfarray(fc_input) / 255.0 * 0.99) + 0.01
    inputs_sig = numpy.array(fc_input, ndmin=2).T

    # перемножаем матрицу весов аксонов на вектор входных данных
    hidden_inputs = numpy.dot(input_hidden_w, inputs_sig)

    # считаем выходы скрытого слоя через функию активации
    hidden_out = fun_active_vector(hidden_inputs)

    # умножаем матрицу весов между скрытым и выходным слоем на вектор выходных значений скрытого слоя
    final_inputs = numpy.dot(hidden_out_w, hidden_out)

    # считаем выходы всей нейронной сети
    final_out = fun_active_vector(final_inputs)
    return final_out


"""
 Главная функция
"""
# рандомом создаем фильтры для сверток
kernel1 = np.random.rand(4, 5, 5) - 0.5
kernel2 = np.random.rand(8, 3, 3) - 0.5


# количество нейронов в полносвязном слое подсчитано заранее
input_nodes, hidden_nodes, out_nodes, learn_node = init_net(800)
input_hidden_w, hidden_out_w = creat_net(input_nodes, hidden_nodes, out_nodes)
(train_X, train_y), (test_X, test_y) = mnist.load_data()

for epoch in range(5):
    print('Epoch#: ', epoch+1)
    for i in range(int(train_X.shape[0]/2)):
        kernel1, kernel2, input_hidden_w, hidden_out_w = train(np.reshape(train_X[i], (1, train_X[i].shape[0],
                                                                                       train_X[i].shape[1])),
                                                                kernel1, kernel2, train_y[i], input_hidden_w,
                                                               hidden_out_w, learn_node)
    test = []
    for j in range(test_X.shape[0]):
        out_session = forward(np.reshape(test_X[j], (1, test_X[j].shape[0], test_X[j].shape[1])), kernel1, kernel2,
                              input_hidden_w, hidden_out_w)
        if test_y[j] == np.argmax(out_session):
            test.append(1)
        else:
            test.append(0)

    print(len(test))
    test = numpy.asarray(test)
    accuracy = (test.sum() / test.size) * 100
    print('Эффективность сети =', accuracy, '%\n')


"""
unit-test для maxpool_backward функции
"""
# y_l = np.array([[
#     [1., 0, 2, 3],
#     [4, 6, 6, 8],
#     [3, 1, 1, 0],
#     [1, 2, 2, 4]
# ]])
# y_l, maxpool_y_l, mask = maxpool(y_l, 2, 2) # макспул работает корректно
# print(maxpool_y_l)
#
# maxpool_y_l_back = maxpool_backward(mask, )