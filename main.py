import numpy as np
import math
import sys
from PyQt5 import QtCore, QtGui, QtWidgets

from gui import Ui_Form  # Подключаем графический интерфейс с файла gui.py


# ---------------------------------------------------------------------------------------
#                 ИНИЦИАЛИЗАЦИЯ ОКНА И ОБЪЯВЛЕНИЕ ГЛОБАЛЬНЫХ ПЕРЕМЕННЫХ
# ---------------------------------------------------------------------------------------

app = QtWidgets.QApplication(sys.argv)
Form = QtWidgets.QWidget()
ui = Ui_Form()
ui.setupUi(Form)
Form.show()

neural_network = None

# ---------------------------------------------------------------------------------------
#                          ОПИСАНИЕ КЛАССА НЕЙРОННОЙ СЕТИ
# ---------------------------------------------------------------------------------------


class ApproxNeuralNetwork(object):  # Класс, описывающий нейронную сеть

    def __init__(self, hidden_nodes_count, a_coeff, b_coeff, c_coeff):  # Конструктор класса ApproxNeuralNetwork

        self.weights_1st_layer = np.random.rand(hidden_nodes_count, 2)  # Задаем случайные значения весам нейронов на первом слое
        self.weights_2nd_layer = np.random.rand(1, hidden_nodes_count)  # Задаем случайные значения весам нейронов на втором слое

        self.a_coeff = a_coeff  # Задаем коэффициенты в квадратичной функции
        self.b_coeff = b_coeff
        self.c_coeff = c_coeff

    def update_graph(self, min_value, max_value):  # Метод вывода значений аппроксимации функции на график
        period = np.arange(min_value, max_value, 0.1)  # Получаем диапазон определения функции
        source_function = self.function(period)  # Получаем значения исходной функции
        approximated_function = []  # Получаем значения аппроксимированой функции
        for x in period:  # Все еще получаем
            approximated_function.append(self.approximate(x))  # Здесь тоже
        approximated_function = np.array(approximated_function)  # Получили

        ui.PlotCanvas.canvas.axes.clear()  # Очищаем полотно графика
        ui.PlotCanvas.canvas.axes.plot(period, source_function)  # Строим исходную функцию
        ui.PlotCanvas.canvas.axes.plot(period, approximated_function) # Строим аппроксимированную функцию
        ui.PlotCanvas.canvas.axes.legend(('Исходная функция', 'Аппроксимация'), loc='upper right')  # Подписываем их
        ui.PlotCanvas.canvas.axes.set_title('Вывод нейросети')  # Подписываем сам график
        ui.PlotCanvas.canvas.draw()  # Обновляем график

    def function(self, x):  # функция которую мы аппроксимируем
        return self.a_coeff * x ** 2 + self.b_coeff * x + self.c_coeff

    def approximate(self, x):  # Метод аппроксимации. Возвращает предсказанное значение функции нейросетью в данной точке
        inputs = np.concatenate((np.array([x]), [1]))  # На вход помимо координаты подаем еще нейрон смещения (всегда единица)
        inputs_hidden_layer = np.dot(self.weights_1st_layer, inputs)  # На вход скрытому слою подаем значения вхожныъ нейронов умноженные на веса
        hidden_layer_output = self.neuron_out_function_vectorized(inputs_hidden_layer)  # На выход скрытого слоя подаем значения нейронов, пройденные через активационную функцию
        total_output = np.dot(self.weights_2nd_layer, hidden_layer_output)  # Уножаем значения выхода скрытого слоя на веса второго слоя

        return total_output.sum()  # На выход подаем сумму значений выходов нейронов скрытого слоя

    def activation_function(self, x):  # Активационная функция нейрона (В нашем случае сигмоида)
        if x >= 250:
            return 1
        elif x <= -250:
            return 0
        else:
            return 1 / (1 + math.e ** (-x))

    def neuron_out_function_vectorized(self, matrix):  # Активационная функция, приведенная в векторный вид
        return np.vectorize(self.activation_function)(matrix)

    def learn(self, min_value, max_value, epoch,
              learn_rate):  # Метод обучения нейросети методом обратного распространения ошибки на заданном отрезке

        learn_rate = np.array([learn_rate])  # Преобразовыве коэффициент обучения в массив из одного числа
        rand_set = np.random.random(epoch) * (max_value - min_value) - max_value   # Случайным образом выбираем количество точек с диапазона равное циклам обучения

        for x in rand_set:  # Перебираем полученные ранее случайные значения

            expected = self.function(x)  # Ожидаемое значение  функции

            inputs = np.concatenate(([x], [1]))  # Также как и в методе approximate делаем предсказание значения
            inputs_hidden_layer = np.dot(self.weights_1st_layer, inputs)
            hidden_layer_outputs = self.neuron_out_function_vectorized(inputs_hidden_layer)
            predicted = np.dot(self.weights_2nd_layer, hidden_layer_outputs)

            error_layer_2 = predicted - expected  # Разница ожидаемого значения решения нейросети от того что получили
            weights_delta_layer_2 = error_layer_2   # Получаем дельту весов второго слоя
            self.weights_2nd_layer -= (np.dot(weights_delta_layer_2, hidden_layer_outputs.reshape(1, len(hidden_layer_outputs)))) * learn_rate  # По формуле умнажаем дальту весов на выход нейронов второго слоя, онимаем получившееся значение, умноженное на коэфф. обучения от значения весов

            error_layer_1 = weights_delta_layer_2 * self.weights_2nd_layer  # Находим ошибку первого слоя нейросети по формуле: дельта весов второго слоя умножить на конечные веса второго слоя
            gradient_layer_1 = hidden_layer_outputs * (1 - hidden_layer_outputs)  # Получаем градиент весов первого слоя по формуле: ошибка * функцияАктивации dx

            weights_delta_layer_1 = error_layer_1 * gradient_layer_1  # Находим дельту весов первого слоя
            self.weights_1st_layer -= np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T * learn_rate  # По формуле умнажаем дальту весов на выход нейронов второго слоя, онимаем получившееся значение, умноженное на коэфф. обучения от значения весов

        self.update_graph(min_value, max_value) # Обновляем график после обучения


# ---------------------------------------------------------------------------------------
#      ПРИВЯЗКА ЭЛЕМЕНТОВ УПРАВЛЕНИЯ ГРАФИЧЕСКОГО ИНТЕРФЕЙСА К МЕТОДАМ НЕЙРОСЕТИ
# ---------------------------------------------------------------------------------------


def initButtonClicked():
    global neural_network
    neural_network = ApproxNeuralNetwork(int(ui.neuronCountEdit.text()), int(ui.aCoeffEdit.text()), int(ui.bCoeffEdit.text()), int(ui.cCoeffEdit.text()))

def learnButtonClicked():
    global neural_network
    if neural_network != None:
        neural_network.learn(int(ui.minXEdit.text()), int(ui.maxXEdit.text()), int(ui.epochCountEdit.text()), float(ui.learnRateEdit.text()))

def approxButtonClicked():
    global neural_network
    if neural_network != None:
        neural_network.update_graph(int(ui.minXEdit.text()), int(ui.maxXEdit.text()))

ui.initButton.clicked.connect(initButtonClicked)
ui.learnButton.clicked.connect(learnButtonClicked)
ui.approxButton.clicked.connect(approxButtonClicked)


sys.exit(app.exec_())
