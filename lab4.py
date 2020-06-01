import math
import numpy as np
from numpy.linalg import solve
from scipy.stats import f, t
from functools import partial
from random import randint
from prettytable import PrettyTable


# *********** Змінні за варіантом ***********
m = 3
N = 8

X1max = 40
X1min = 10
X2max = 35
X2min = -15
X3max = 5
X3min = -15

Xmax_average = (X1max + X2max + X3max) / 3
Xmin_average = (X1min + X2min + X3min) / 3

y_max = round(200 + Xmax_average)
y_min = round(200 + Xmin_average)

# *********** *********** *********** ***********

def kriteriy_fishera(y, y_aver, y_new, n, m, d):
    """
    Функція для знаходження критерія фішера
    """ 
    S_kv_ad = (m / (n - d)) * sum([(y_new[i] - y_aver[i])**2 for i in range(len(y))])
    S_kv_b = s_kv(y, y_aver, n, m)
    S_kv_b_aver = sum(S_kv_b) / n

    return S_kv_ad / S_kv_b_aver


def s_kv(y, y_aver, n, m):
    """
    Функція для знаходження квадратної дисперсії
    """
    res = []
    for i in range(n):
        s = sum([(y_aver[i] - y[i][j])**2 for j in range(m)]) / m
        res.append(s)
    return res


def getDispersion(y_rows) -> list:
    disp_array = [0] * 8 

    for k in range(len(disp_array)):
        for i in range(m):
            disp_array[k] += ((np.average(y_rows[k]) - y_rows[k][i])**2) / m

    return disp_array


def getCodableX(factorsArray):
    x1, x2, x3 = [], [], []
    for factorNumber in range(len(factorsArray)):
        factor = factorsArray[factorNumber]
        for i in factor:
            if i > 0:
                if factorNumber == 0:
                    x1.append(X1max)
                elif factorNumber == 1:
                    x2.append(X2max)
                else:
                    x3.append(X3max)    
            else:
                if factorNumber == 0:
                    x1.append(X1min)
                elif factorNumber == 1:
                    x2.append(X2min)
                else:
                    x3.append(X3min)  
    return x1, x2, x3


def getRandomY():
    y1, y2, y3 = [], [], []

    for _ in range(0, 8):
        y1.append(randint(y_min, y_max))
        y2.append(randint(y_min, y_max))
        y3.append(randint(y_min, y_max))

    return y1, y2, y3


def getYRows(y1, y2, y3):
    y_rows = []
    for i in range(8):
        y_rows.append([y1[i], y2[i], y3[i]])
    
    return y_rows


def getYAverage(y_rows) -> list:
    y_average_all = []

    for i in range(len(y_rows)):
        y_average_all.append(np.average(y_rows[i]))

    return y_average_all


def cohren(f1, f2, q=0.05):
    q1 = q / f1
    fisher_value = f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)
    return fisher_value / (fisher_value + f1 - 1)


def plan_matrix(n, m):
    """
    Функція для знаходження матриці планування
    """
    y = np.zeros(shape=(n,m))
    for i in range(n):
        for j in range(m):
            y[i][j] = randint(y_min, y_max)

    x_norm = np.array([[1, -1, -1, -1],
                       [1, -1,  1,  1],
                       [1,  1, -1,  1],
                       [1,  1,  1, -1],
                       [1, -1, -1,  1],
                       [1, -1,  1, -1],
                       [1,  1, -1, -1],
                       [1,  1,  1,  1]])
    x_norm = x_norm[:len(y)]

    x_range = [(X1min, X1max), (X1min, X1max), (X1min, X1max)]
    x = np.ones(shape=(len(x_norm), len(x_norm[0])))
    for i in range(len(x_norm)):
        for j in range(1, len(x_norm[i])):
            if x_norm[i][j] == -1:
                x[i][j] = x_range[j-1][0]
            else:
                x[i][j] = x_range[j-1][1]

    print('\nМатриця планування')
    matrix = np.concatenate((x,y),axis=1)
    table = PrettyTable()
    
    yFieldNames = []
    yFieldNames += (f'Y{i+1}' for i in range(m))

    fieldNames = ["X0c", "X1c", "X2c", "X3c"] + yFieldNames
    table.field_names = fieldNames

    for i in range(len(matrix)):
        table.add_row(matrix[i])

    print(table)

    return x, y


def find_coefficient(x, y_aver, n):
    """
    Функція для знаходження коефіціентів B
    """
    mx1 = sum(x[:, 1]) / n
    mx2 = sum(x[:, 2]) / n
    mx3 = sum(x[:, 3]) / n
    my = sum(y_aver) / n

    a12 = sum([x[i][1] * x[i][2] for i in range(len(x))]) / n
    a13 = sum([x[i][1] * x[i][3] for i in range(len(x))]) / n
    a23 = sum([x[i][2] * x[i][3] for i in range(len(x))]) / n
    a11 = sum([i ** 2 for i in x[:, 1]]) / n
    a22 = sum([i ** 2 for i in x[:, 2]]) / n
    a33 = sum([i ** 2 for i in x[:, 3]]) / n
    a1 = sum([y_aver[i] * x[i][1] for i in range(len(x))]) / n
    a2 = sum([y_aver[i] * x[i][2] for i in range(len(x))]) / n
    a3 = sum([y_aver[i] * x[i][3] for i in range(len(x))]) / n

    X = [[  1, mx1, mx2, mx3],
         [mx1, a11, a12, a13], 
         [mx2, a12, a22, a23], 
         [mx3, a13, a23, a33]]

    Y = [my, a1, a2, a3]
    # Вирішимо систему рівнянь для коефіціентів
    B = [round(i, 2) for i in solve(X, Y)]
    print('\nРівняння регресії')
    print(f'{B[0]} + {B[1]}*x1 + {B[2]}*x2 + {B[3]}*x3')

    return B


def main(m, effectVzaemodiy = False, isSetEffectVzaemodiy = False):
    N = 8
    fisher = partial(f.ppf, q=1-0.05)
    student = partial(t.ppf, q=1-0.025)

    # y
    y1, y2, y3 = getRandomY()

    # y_rows
    y_rows = getYRows(y1, y2, y3)

    # середнє значення y_rows
    y_average = getYAverage(y_rows)

    disp_list = []

    list_bi = []

    if effectVzaemodiy:
        # З Ефектом взаємодії
        
        x0_factor = [ 1,  1,  1,  1,  1,  1,  1,  1]
        x1_factor = [-1, -1,  1,  1, -1, -1,  1,  1]
        x2_factor = [-1,  1, -1,  1, -1,  1, -1,  1]
        x3_factor = [-1,  1,  1, -1,  1, -1, -1,  1]

        x1x2_factor = [a*b for a, b in zip(x1_factor, x2_factor)]
        x1x3_factor = [a*b for a, b in zip(x1_factor, x3_factor)]
        x2x3_factor = [a*b for a, b in zip(x2_factor, x3_factor)]
        x1x2x3_factor = [a*b*c for a, b, c in zip(x1_factor, x2_factor, x3_factor)]

        # Кодовані значення
        x1, x2, x3 = getCodableX([x1_factor, x2_factor, x3_factor])
        
        x1x2 = [a*b for a, b in zip(x1, x2)]
        x1x3 = [a*b for a, b in zip(x1, x3)]
        x2x3 = [a*b for a, b in zip(x2, x3)]
        x1x2x3 = [a*b*c for a, b, c in zip(x1, x2, x3)]

        list_for_solve_b = [x0_factor, x1_factor, x2_factor, x3_factor, x1x2_factor, x1x3_factor, x2x3_factor, x1x2x3_factor]
        list_for_solve_a = list(zip(x0_factor, x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3))

        # Коефіцієнти регресії для нормованих значень факторів коефіцієнтів регресії для нормованих значень факторів
        list_bi = []
        for k in range(N):
            S = 0
            for i in range(N):
                S += (list_for_solve_b[k][i] * y_average[i]) / N
            list_bi.append(round(S, 5))

        # Список з дисперсіями
        disp_list = getDispersion(y_rows)

        # Округлені значення дисперсії та yAverage для таблиці
        dispListToTable = [round(i, 3) for i in disp_list] 
        yAverageToTable = [round(i, 3) for i in y_average]

        # Створюємо 2 таблиці    
        pt1 = PrettyTable()
        pt2 = PrettyTable()

        # Данні для наповняння таблиці
        column_names = ["X0", "X1", "X2", "X3", "X1X2", "X1X3", "X2X3", "X1X2X3", "Y1", "Y2", "Y3", "Y_aver", "S^2"]
        column_values1 = [x0_factor, x1_factor, x2_factor, x3_factor, x1x2_factor, x1x3_factor, x2x3_factor, x1x2x3_factor, y1, y2, y3, yAverageToTable, dispListToTable]
        column_values2 = [x0_factor, x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3, y1, y2, y3, yAverageToTable, dispListToTable]

        # Наповнюємо першу таблицю
        for i in range(len(column_names)):
            pt1.add_column(column_names[i], column_values1[i])

        # Виводимо першу таблицю
        print(pt1, "\n")

        print("y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3 \n".format(list_bi[0], list_bi[1],
                                                                                                list_bi[2], list_bi[3],
                                                                                                list_bi[4], list_bi[5],
                                                                                                list_bi[6], list_bi[7]))                                                                                  # list_bi[2], list_bi[3]))

        # Наповнюємо другу таблицю
        for i in range(len(column_names)):
            pt2.add_column(column_names[i], column_values2[i])

        # Виводимо педругуршу таблицю
        print(pt2, '\n')

        # Визначення коефіцієнтів рівняння регресії ПФЕ
        list_ai = [round(i, 5) for i in solve(list_for_solve_a, y_average)]

        print("y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3".format(list_ai[0], list_ai[1], list_ai[2], list_ai[3], list_ai[4], list_ai[5], list_ai[6], list_ai[7]))                                                     
    
    else:
        # Без ефекту взаємодії
        x, y = plan_matrix(N, m)
        list_bi = find_coefficient(x, y_average, N)
        disp_list = getDispersion(y)

    # Теоретичне
    Gp = max(disp_list) / sum(disp_list)
    
    F1 = m-1
    N = len(y1)
    F2 = N

    # Табличне
    Gt = cohren(F1, F2)
    print("\nGp = ", Gp, " Gt = ", Gt)

    if Gp < Gt:
        print("Gp < Gt -> Дисперсія однорідна!\n")
        
        Dispersion_B = sum(disp_list) / N
        Dispersion_beta = Dispersion_B / (m * N)
        S_beta = math.sqrt(abs(Dispersion_beta))

        t_list = []

        for i in range(len(list_bi)):
            t_list.append(abs(list_bi[i]) / S_beta)

        F3 = F1 * F2
        d = 0
        T = student(df=F3)
        print("t стьюдента табличне = ", T)

        for i in range(len(t_list)):
            if t_list[i] < T:
                print("Коефіціент {} не є значущим, виключаємо його".format(t_list[i]))
                list_bi[i] = 0

            else:
                d += 1
        
        # Критерії які підходять
        Y_counted_for_Student = [] 

        for i in range(8):
            if effectVzaemodiy:
                Y_counted_for_Student.append(list_bi[0] + list_bi[1] * x1[i] + list_bi[2] * x2[i] + list_bi[3] * x3[i] + list_bi[4] * x1x2[i] \
                + list_bi[5]*x1x3[i] + list_bi[6]*x2x3[i] + list_bi[7]*x1x2x3[i])
            else: 
                Y_counted_for_Student.append(list_bi[0] + list_bi[1] * x[i][1] + list_bi[2] * x[i][2] + list_bi[3] * x[i][3])

        # Табличне
        F4 = N - d
        Ft = fisher(dfn=F4, dfd=F3)

        # Практичне
        Fp = kriteriy_fishera(y_rows, y_average, Y_counted_for_Student, N, m, d)

        effectVzaemodiy = True

        if Fp > Ft:
            print("Модель не адекватна")
            m = 3
            
            # Якщо вже було спробувано ефект взаємодії повернутися на початок
            if isSetEffectVzaemodiy:
                main(m, False, False)
            else:
                print("Спробуємо ефект взаємодії")
                main(m, effectVzaemodiy, True)

        else:   
            print("МОДЕЛЬ АДЕКВАТНА")


    else:
        print("Дисперсія неоднорідна. Спробуємо з m = {}".format(m + 1))        
        m += 1
        main(m, effectVzaemodiy, isSetEffectVzaemodiy)


if __name__ == "__main__":
    main(m)
