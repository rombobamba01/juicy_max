import math
import numpy as np
from numpy.linalg import solve
from scipy.stats import f, t
from functools import partial
import random as r
from prettytable import PrettyTable
import sklearn.linear_model as lm
from datetime import datetime
import time

start_time = datetime.now()


# *********** Змінні за варіантом ***********
np.set_printoptions(suppress=True)
m = 3
N = 8


X1max = 15
X1min = -5
X2max = 10
X2min = -35
X3max = -10
X3min = -35

maximumIteration = 100

Xmax_average = (X1max + X2max + X3max) / 3
Xmin_average = (X1min + X2min + X3min) / 3

y_max = round(200 + Xmax_average)
y_min = round(200 + Xmin_average)

# *********** *********** ***********


def getMatrix(n, m, typeMatrix = 0):
    """
    Функція для знаходження нормованої та натуралізованої матриці та планування
    typeMatrix = 0 (лінійна форма) (n = 8)
    typeMatrix = 1 (рівняня з ефектом взаємодії) (n = 8)
    typeMatrix = 2 (рівняня з ефектом взаємодії та квадратних членів) (n = 14)
    """

    # Список з значеннями X
    x_range = [(X1min, X1max), (X2min, X2max), (X3min, X3max)]

    # Поля Y для таблиць
    yFieldNames = [f"Y{i+1}" for i in range(m)]
    yFieldNames.append("Y_av")

    # ****** ****** ****** Лінійна форма ****** ****** ******
    # Матриця нормованих значень для typeMatrix = 0
    array_standart = [[1, -1, -1, -1],
                      [1, -1,  1,  1],
                      [1,  1, -1,  1],
                      [1,  1,  1, -1],
                      [1, -1, -1,  1],
                      [1, -1,  1, -1],
                      [1,  1, -1, -1],
                      [1,  1,  1,  1]]
    x_norm_standart = np.array(array_standart)

    # Матриця натуральних значень для typeMatrix = 0
    x_nat_standart = np.ones(shape=(len(x_norm_standart), len(x_norm_standart[0])))
    for i in range(len(x_norm_standart)):
        for j in range(1, len(x_norm_standart[i])):
            if x_norm_standart[i][j] == -1:
                x_nat_standart[i][j] = x_range[j-1][0]
            else:
                x_nat_standart[i][j] = x_range[j-1][1]
    # ****** ****** ****** ******


    # ****** ****** ****** Ефект взаємодії ****** ****** ******
    # Матриця нормованих значень для typeMatrix = 1
    array_vzaemodia = array_standart
    for row in array_vzaemodia:
        row.append(row[1]*row[2])
        row.append(row[1]*row[3])
        row.append(row[2]*row[3])
        row.append(row[1]*row[2]*row[3])
    x_norm_vzaemodia = np.array(array_vzaemodia)

    # Матриця натуральних значень для typeMatrix = 1
    x_nat_vzaemodia = np.ones(shape=(len(x_norm_standart), 4))
    for i in range(len(x_norm_vzaemodia)):
        x_nat_vzaemodia[i][0] = x_norm_standart[i][1] * x_norm_standart[i][2]
        x_nat_vzaemodia[i][1] = x_norm_standart[i][1] * x_norm_standart[i][3]
        x_nat_vzaemodia[i][2] = x_norm_standart[i][2] * x_norm_standart[i][3]
        x_nat_vzaemodia[i][3] = x_norm_standart[i][1] * x_norm_standart[i][2] * x_norm_standart[i][3]
    # ****** ****** ****** ******


    # ****** ****** ****** Ефект взаємодії + квадратичних членів ****** ****** ******
    # Матриця нормованих значень для typeMatrix = 2
    array_kv = array_vzaemodia
    l = 1.73
    array_kv.append([1, -l, 0, 0, 0, 0, 0, 0])
    array_kv.append([1, l, 0, 0, 0, 0, 0, 0])
    array_kv.append([1, 0, -l, 0, 0, 0, 0, 0])
    array_kv.append([1, 0, l, 0, 0, 0, 0, 0])
    array_kv.append([1, 0, 0, -l, 0, 0, 0, 0])
    array_kv.append([1, 0, 0, l, 0, 0, 0, 0])

    for row in array_kv:
        row.append(round(row[1]*row[1], 4))
        row.append(round(row[2]*row[2], 4))
        row.append(round(row[3]*row[3], 4))
    x_norm_kv = np.array(array_kv)

    # Матриця натуральних значень для typeMatrix = 2
    # x_nat_kv_1 це матриця n = 14 де стовпці x0, x1, x2, x3
    x_nat_kv_1 = np.ones(shape=(len(x_norm_kv), len(x_norm_standart[0])))

    for i in range(len(x_norm_kv)):
        for j in range(1, 4):
            if x_norm_kv[i][j] == -1:
                # При нормованому -1
                x_nat_kv_1[i][j] = x_range[j-1][0]
            elif x_norm_kv[i][j] == 1:
                # При нормованому 1
                x_nat_kv_1[i][j] = x_range[j-1][1]
            elif abs(x_norm_kv[i][j]) > 1:
                # При l
                x0 = (x_range[j-1][0] + x_range[j-1][1]) / 2
                xDelta = x_range[j-1][1] - x0
                x_nat_kv_1[i][j] = round(x_norm_kv[i][j] * xDelta + x0, 2)
            else:
                x0 = (x_range[j-1][0] + x_range[j-1][1]) / 2
                x_nat_kv_1[i][j] = round(x0, 2)

    # x_nat_kv_2 це матриця n = 14 для значення еффекту взаємодії та квадратних членів
    x_nat_kv_2 = np.ones(shape=(len(x_norm_kv), 7))
    for i in range(len(x_nat_kv_1)):
        x_nat_kv_2[i][0] = round(x_nat_kv_1[i][1] * x_nat_kv_1[i][2], 2)
        x_nat_kv_2[i][1] = round(x_nat_kv_1[i][1] * x_nat_kv_1[i][3], 2)
        x_nat_kv_2[i][2] = round(x_nat_kv_1[i][2] * x_nat_kv_1[i][3], 2)
        x_nat_kv_2[i][3] = round(x_nat_kv_1[i][1] * x_nat_kv_1[i][2] * x_nat_kv_1[i][3], 2)

        x_nat_kv_2[i][4] = round(x_nat_kv_1[i][1] * x_nat_kv_1[i][1], 2)
        x_nat_kv_2[i][5] = round(x_nat_kv_1[i][2] * x_nat_kv_1[i][2], 2)
        x_nat_kv_2[i][6] = round(x_nat_kv_1[i][3] * x_nat_kv_1[i][3], 2)
    # ****** ****** ****** ******


    # Згенеровані значення Y для N = 14
    yFull = np.zeros(shape=(14,m))
    for i in range(14):
        for j in range(m):
            x1 = x_nat_kv_1[i][1]
            x2 = x_nat_kv_1[i][2]
            x3 = x_nat_kv_1[i][3]
            y = myVariantFunction(x1, x2, x3, typeMatrix)
            yFull[i][j] = y + r.randrange(0, 10) - 5

    # Згенеровані значення Y для N = 8
    y = yFull[:8]

    # Середні значення Y
    y_average = getYAverage(y)
    y_average_Full = getYAverage(yFull)

    # Середні значення Y для таблиці [[],[]]
    table_y_average = []
    for i in range(len(y_average)):
        table_y_average.append([y_average[i]])

    table_y_average_full = []
    for i in range(len(y_average_Full)):
        table_y_average_full.append([y_average_Full[i]])


    if typeMatrix == 0:
        fieldNames = ["X0", "X1", "X2", "X3"]
        printMatrix("Нормована матриця планування лінійної форми", x_norm_standart, fieldNames)
        fieldNames += yFieldNames
        printMatrix("Натуралізована матриця планування лінійної форми", np.concatenate((x_nat_standart, y, table_y_average), axis=1), fieldNames)
        return x_norm_standart, x_nat_standart, y
    elif typeMatrix == 1:
        fieldNames = ["X0", "X1", "X2", "X3", "X12", "X13", "X23", "X123"]
        printMatrix("Нормована матриця планування з ефектом взаємодії", x_norm_vzaemodia, fieldNames)
        fieldNames += yFieldNames
        printMatrix("Натуралізована матриця планування з ефектом взаємодії", np.concatenate((x_nat_standart, x_nat_vzaemodia, y, table_y_average), axis=1), fieldNames)
        return x_norm_vzaemodia, np.concatenate((x_nat_standart, x_nat_vzaemodia), axis=1), y
    else:
        fieldNames = ["X0", "X1", "X2", "X3", "X12", "X13", "X23", "X123", "X1^2", "X2^2", "X3^2"]
        printMatrix("Нормована матриця планування з ефектом взаємодії та квадратних коренів", x_norm_kv, fieldNames)
        fieldNames += yFieldNames
        printMatrix("Натуралізована матриця планування з ефектом взаємодії та квадратних коренів", np.concatenate((x_nat_kv_1, x_nat_kv_2, yFull, table_y_average_full), axis=1), fieldNames)
        return x_norm_kv, np.concatenate((x_nat_kv_1, x_nat_kv_2), axis=1), yFull


def printMatrix(name, values, fields):
    """
    Функція для виведення таблиць
    """
    print("\n", name)
    table = PrettyTable()
    table.field_names = fields

    for i in range(len(values)):
        table.add_row(values[i])

    print(table)


def s_kv(y, y_aver, n, m):
    """
    Функція для знаходження квадратної дисперсії
    """
    res = []
    for i in range(n):
        s = sum([(y_aver[i] - y[i][j])**2 for j in range(m)]) / m
        res.append(s)
    return res


def kriteriy_fishera(y, y_aver, y_new, n, m, d):
    """
    Функція для знаходження критерія фішера
    """
    S_kv_ad = (m / (n - d)) * sum([(y_new[i] - y_aver[i])**2 for i in range(len(y))])
    S_kv_b = s_kv(y, y_aver, n, m)
    S_kv_b_aver = sum(S_kv_b) / n

    return S_kv_ad / S_kv_b_aver



def cohren(f1, f2, q=0.05):
    """
    Функція для знаходження критерія кохрена
    """
    q1 = q / f1
    fisher_value = f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)
    return fisher_value / (fisher_value + f1 - 1)


def getYAverage(y):
    """
    Функція для знаходження середнього значення Y
    """
    y_average = []
    for row in y:
        y_average.append(round(sum(row) / 3, 3))

    return y_average
    

def myVariantFunction(x1, x2, x3, typeMatrix) -> float:
    """
    207 variant
    """
    result = 0
    if typeMatrix == 0:
        # result = 4.6 + 6.1 * x1 + 9.6 * x2 + 1.2 * x3
        result = 4.6 + 6.1 * x1 + 9.6 * x2 + 1.2 * x3 + (0.3 * x1 * x1) + (0.6 * x2 * x2) + (3.8 * x3 * x3) + (7.7 * x1 * x2) + (0.4 * x1 * x3) + (5.3 * x2 * x3) + (7.1 * x1 * x2 * x3)

    else:
        result = 4.6 + 6.1 * x1 + 9.6 * x2 + 1.2 * x3 + (0.3 * x1 * x1) + (0.6 * x2 * x2) + (3.8 * x3 * x3) + (7.7 * x1 * x2) + (0.4 * x1 * x3) + (5.3 * x2 * x3) + (7.1 * x1 * x2 * x3)
    result = round(result, 2)
    return result


def main(m, iteration, effectVzaemodiyAndKv = False, isSetEffectVzaemodiyAndKv = False):
    if iteration == maximumIteration:
        print("За 100 ітерацій матриця не стала адекватною")
        return
    else:
        iteration += 1 

    fisher = partial(f.ppf, q=1-0.05)
    student = partial(t.ppf, q=1-0.025)

    # Значення для лінійної форми
    matrixType = 0
    N = 8
    if effectVzaemodiyAndKv == True:
        # Значення для рівняня з ефектом взаємодії та квадратних членів
        matrixType = 2
        N = 14

    # Нормовані, натуралізовані значенння X, та Y
    _, x_nat, y = getMatrix(N, m, matrixType)

    # Середнє значення Y
    y_average = getYAverage(y)

    
    list_bi = np.linalg.lstsq(x_nat, y_average, rcond=None)[0]


    y_perevirka = [] 

    for i in range(N):
        if effectVzaemodiyAndKv:
            y_perevirka.append(list_bi[0] + list_bi[1] * x_nat[i][1] + list_bi[2] * x_nat[i][2] + list_bi[3] * x_nat[i][3] + list_bi[4] * x_nat[i][4] \
            + list_bi[5]* x_nat[i][5] + list_bi[6] * x_nat[i][6] + list_bi[7] * x_nat[i][7] + list_bi[8] * x_nat[i][8] + list_bi[9] * x_nat[i][9] + list_bi[10] * x_nat[i][10])
        else: 
            y_perevirka.append(list_bi[0] + list_bi[1] * x_nat[i][1] + list_bi[2] * x_nat[i][2] + list_bi[3] * x_nat[i][3])

    for i in range(len(y_perevirka)):
        print(" y{} (перевірка) = {} ≈ {} ".format((i+1), y_perevirka[i], y_average[i]))

    # Массив значення дисперсії
    disp_list = dispersion(y, y_average, N, m)

    # Теоретичне значення перевірки кохрена
    Gp = max(disp_list) / sum(disp_list)
    
    F1 = m-1
    F2 = N

    # Табличне значення перевірки кохрена
    Gt = cohren(F1, F2)
    print("\nGp = ", Gp, " Gt = ", Gt)

    if Gp < Gt:
        print("Оскільки Gp < Gt, то Дисперсія однорідна!\n")
        
        Dispersion_B = sum(disp_list) / N
        Dispersion_beta = Dispersion_B / (m * N)
        S_beta = math.sqrt(abs(Dispersion_beta))
        t_list = []
        for i in range(len(list_bi)):
            t_list.append(abs(list_bi[i]) / S_beta)

        # d - кількість критерів рівняння що залишилось
        d = 0

        F3 = F1 * F2
        T = student(df=F3)

        print("t стьюдента табличне = ", T)

        for i in range(len(t_list)):
            if t_list[i] < T:
                print("Коефіціент {} не є значущим, виключаємо його".format(t_list[i]))
                list_bi[i] = 0
            else:
                d += 1

        print("\nB: ", list(list_bi))

        print("")
        # Критерії які підходять
        Y_counted_for_Student = [] 

        for i in range(N):
            if effectVzaemodiyAndKv:
                Y_counted_for_Student.append(list_bi[0] + list_bi[1] * x_nat[i][1] + list_bi[2] * x_nat[i][2] + list_bi[3] * x_nat[i][3] + list_bi[4] * x_nat[i][4] \
                + list_bi[5]* x_nat[i][5] + list_bi[6] * x_nat[i][6] + list_bi[7] * x_nat[i][7] + list_bi[8] * x_nat[i][8] + list_bi[9] * x_nat[i][9] + list_bi[10] * x_nat[i][10])
            else: 
                Y_counted_for_Student.append(list_bi[0] + list_bi[1] * x_nat[i][1] + list_bi[2] * x_nat[i][2] + list_bi[3] * x_nat[i][3])

        print("Значення Y після перевірки значимості коефіцієнтів:")
        for i in range(len(y_perevirka)):
            print(" y{} = {} ≈ {} (delta = {})".format((i+1), Y_counted_for_Student[i], y_average[i], Y_counted_for_Student[i] - y_average[i] ))

        # Табличне значення перевірки фішера
        F4 = N - d
        Ft = fisher(dfn=F4, dfd=F3)

        # Практичне значення перевірки фішера
        Fp = kriteriy_fishera(y, y_average, Y_counted_for_Student, N, m, d)

        effectVzaemodiyAndKv = True
        if Fp > Ft:
            print("Модель не адекватна Fp = ", Fp, "  Ft = ", Ft)
            m = 3
            
            if isSetEffectVzaemodiyAndKv:
                # Якщо вже було спробувано ефект взаємодії та квадратних членів повернутися на початок
                main(m, iteration, False, False)
            else:
                print("\nСпробуємо ефект взаємодії + квадратних членів")
                main(m, iteration, effectVzaemodiyAndKv, True)
        else:   
            print("Модель адекватна Fp = ", Fp, " < Ft = ", Ft)

            # print("МОДЕЛЬ АДЕКВАТНА")

    else:
        print("Дисперсія неоднорідна. Спробуємо з m = {}".format(m + 1))        
        m += 1
        main(m, iteration, effectVzaemodiyAndKv, isSetEffectVzaemodiyAndKv)

def get_dispersion(y_aver, y):
    return sum([(i-y_aver)**2 for i in y])/len(y)   

def dispersion(y, y_aver, n, m):
    res = []
    for i in range(n):
        s = sum([(y_aver[i] - y[i][j]) ** 2 for j in range(m)]) / m
        res.append(round(s, 3))
    return res  

if __name__ == "__main__":
    main(m, 0)
 print(datetime.now() - start_time)
