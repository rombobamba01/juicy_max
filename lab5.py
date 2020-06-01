import math
import numpy as np
from numpy.linalg import solve
from scipy.stats import f, t
from functools import partial
from random import randint
from prettytable import PrettyTable
import sklearn.linear_model as lm


# *********** Змінні за варіантом ***********
np.set_printoptions(suppress=True)
m = 3
N = 8

X1max = 8
X1min = -5
X2max = 4
X2min = -7
X3max = 4
X3min = -10

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
    typeMatrix = 2 (рівняня з ефектом взаємодії та квадратних членів) (n = 15)
    """

    # Список з значеннями X
    x_range = [(X1min, X1max), (X2min, X2max), (X3min, X3max)]

    # Поля Y для таблиць
    yFieldNames = [f"Y{i+1}" for i in range(m)]

    # Згенеровані значення Y для N = 15
    yFull = np.zeros(shape=(15,m))
    for i in range(15):
        for j in range(m):
            yFull[i][j] = randint(y_min, y_max)

    # Згенеровані значення Y для N = 8
    y = yFull[:8]


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
    array_kv.append([1, -1.215, 0, 0, 0, 0, 0, 0])
    array_kv.append([1, 1.215, 0, 0, 0, 0, 0, 0])
    array_kv.append([1, 0, -1.215, 0, 0, 0, 0, 0])
    array_kv.append([1, 0, 1.215, 0, 0, 0, 0, 0])
    array_kv.append([1, 0, 0, -1.215, 0, 0, 0, 0])
    array_kv.append([1, 0, 0, 1.215, 0, 0, 0, 0])
    array_kv.append([1, 0, 0, 0, 0, 0, 0, 0])

    for row in array_kv:
        row.append(round(row[1]*row[1], 4))
        row.append(round(row[2]*row[2], 4))
        row.append(round(row[3]*row[3], 4))
    x_norm_kv = np.array(array_kv)

    # Матриця натуральних значень для typeMatrix = 2
    # x_nat_kv_1 це матриця n = 15 де стовпці x0, x1, x2, x3
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

    # x_nat_kv_1 це матриця n = 15 для значення еффекту взаємодії та квадратних членів
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


    if typeMatrix == 0:
        fieldNames = ["X0", "X1", "X2", "X3"] + yFieldNames
        printMatrix("Нормована матриця планування лінійної форми", np.concatenate((x_norm_standart, y), axis=1), fieldNames)
        printMatrix("Натуралізована матриця планування лінійної форми", np.concatenate((x_nat_standart, y), axis=1), fieldNames)
        return x_norm_standart, x_nat_standart, y
    elif typeMatrix == 1:
        fieldNames = ["X0", "X1", "X2", "X3", "X12", "X13", "X23", "X123"] + yFieldNames
        printMatrix("Нормована матриця планування з ефектом взаємодії", np.concatenate((x_norm_vzaemodia, y), axis=1), fieldNames)
        printMatrix("Натуралізована матриця планування з ефектом взаємодії", np.concatenate((x_nat_standart, x_nat_vzaemodia, y), axis=1), fieldNames)
        return x_norm_vzaemodia, np.concatenate((x_nat_standart, x_nat_vzaemodia), axis=1), y
    else:
        fieldNames = ["X0", "X1", "X2", "X3", "X12", "X13", "X23", "X123", "X1^2", "X2^2", "X3^2"] + yFieldNames
        printMatrix("Нормована матриця планування з ефектом взаємодії та квадратних коренів", np.concatenate((x_norm_kv, yFull), axis=1), fieldNames)
        printMatrix("Натуралізована матриця планування з ефектом взаємодії та квадратних коренів", np.concatenate((x_nat_kv_1, x_nat_kv_2, yFull), axis=1), fieldNames)
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


def find_coef(X, Y, norm=False):
    """
    Функція для знаходження коефіціентів рівняння
    """
    skm = lm.LinearRegression(fit_intercept=False)
    skm.fit(X, Y)
    B = skm.coef_

    if norm == 1:
        print('\nКоефіцієнти рівняння регресії з нормованими X:')
    else:
        print('\nКоефіцієнти рівняння регресії:')
    B = [round(i, 3) for i in B]
    print(B)
    return B


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


def getDispersion(y_rows) -> list:
    """
    Функція для знаходження дисперсії
    """
    disp_array = [0] * 8 
    for k in range(len(disp_array)):
        for i in range(m):
            disp_array[k] += ((np.average(y_rows[k]) - y_rows[k][i])**2) / m

    return disp_array


def getYAverage(y):
    """
    Функція для знаходження середнього значення Y
    """
    y_average = []
    for row in y:
        y_average.append(sum(row) / 3)

    return y_average


def main(m, effectVzaemodiyAndKv = False, isSetEffectVzaemodiyAndKv = False):
    fisher = partial(f.ppf, q=1-0.05)
    student = partial(t.ppf, q=1-0.025)

    # Значення для лінійної форми
    matrixType = 0
    N = 8
    if effectVzaemodiyAndKv == True:
        # Значення для рівняня з ефектом взаємодії та квадратних членів
        matrixType = 2
        N = 15

    # Нормовані, натуралізовані значенння X, та Y
    x_norm, x_nat, y = getMatrix(N, m, matrixType)

    # Середнє значення Y
    y_average = getYAverage(y)

    # Коефіціенти нормованого рівняння
    find_coef(x_norm, y_average, norm = True)

    # Коефіціенти натуралізованого рівняння
    list_bi = find_coef(x_nat, y_average, norm = False)

    # Массив значення дисперсії
    disp_list = getDispersion(y)

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
        
        print("")
        # Критерії які підходять
        Y_counted_for_Student = [] 

        for i in range(N):
            if effectVzaemodiyAndKv:
                Y_counted_for_Student.append(list_bi[0] + list_bi[1] * x_nat[i][1] + list_bi[2] * x_nat[i][2] + list_bi[3] * x_nat[i][3] + list_bi[4] * x_nat[i][4] \
                + list_bi[5]* x_nat[i][5] + list_bi[6] * x_nat[i][6] + list_bi[7] * x_nat[i][5] + list_bi[8] * x_nat[i][6] + list_bi[8] * x_nat[i][7])
            else: 
                Y_counted_for_Student.append(list_bi[0] + list_bi[1] * x_nat[i][1] + list_bi[2] * x_nat[i][2] + list_bi[3] * x_nat[i][3])

        # Табличне значення перевірки фішера
        F4 = N - d
        Ft = fisher(dfn=F4, dfd=F3)

        # Практичне значення перевірки фішера
        Fp = kriteriy_fishera(y, y_average, Y_counted_for_Student, N, m, d)

        effectVzaemodiyAndKv = True
        if Fp > Ft:
            print("Модель не адекватна   Fp = ", Fp, " > Ft = ", Ft)
            m = 3
            
            if isSetEffectVzaemodiyAndKv:
                # Якщо вже було спробувано ефект взаємодії та квадратних членів повернутися на початок
                main(m, False, False)
            else:
                print("\nСпробуємо ефект взаємодії + квадратних членів")
                main(m, effectVzaemodiyAndKv, True)
        else:   
            print("МОДЕЛЬ АДЕКВАТНА")

    else:
        print("Дисперсія неоднорідна. Спробуємо з m = {}".format(m + 1))        
        m += 1
        main(m, effectVzaemodiyAndKv, isSetEffectVzaemodiyAndKv)


if __name__ == "__main__":
    main(m)
