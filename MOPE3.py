import numpy as np
import scipy.stats

x1min = -5
x1max = 15
x2min = -35
x2max = 10
x3min = -35
x3max = -10

mx_max = (x1max + x2max + x3max) / 3
mx_min = (x1min + x2min + x3min) / 3
y_max = mx_max + 200
y_min = mx_min + 200

y_list = np.random.randint(y_min, y_max, (4, 3))  # create tab 4*3 random 'y' in [y_min; y_max]

x_matrix = [
    [x1min, x2min, x3min],
    [x1min, x2max, x3max],
    [x1max, x2min, x3max],
    [x1max, x2max, x3min]
]

my_list = []
mx1 = 0
mx2 = 0
mx3 = 0
for obj in y_list:
    my_list.append((obj[0]+obj[1]+obj[2])/3)

for obj in x_matrix:
    mx1 += obj[0]
    mx2 += obj[1]
    mx3 += obj[2]
my1 = my_list[0]
my2 = my_list[1]
my3 = my_list[2]
my4 = my_list[3]
mx1 /= 4
mx2 /= 4
mx3 /= 4
my = (my1 + my2 + my3 + my4)/4

"""Coefficients"""
a1 = (x_matrix[0][0]*my1 + x_matrix[1][0]*my2 + x_matrix[2][0]*my3 + x_matrix[3][0]*my4)/4
a2 = (x_matrix[0][1]*my1 + x_matrix[1][1]*my2 + x_matrix[2][1]*my3 + x_matrix[3][1]*my4)/4
a3 = (x_matrix[0][2]*my1 + x_matrix[1][2]*my2 + x_matrix[2][2]*my3 + x_matrix[3][2]*my4)/4
a11 = (x_matrix[0][0]**2 + x_matrix[1][0]**2 + x_matrix[2][0]**2 + x_matrix[3][0]**2)/4
a22 = (x_matrix[0][1]**2 + x_matrix[1][1]**2 + x_matrix[2][1]**2 + x_matrix[3][1]**2)/4
a33 = (x_matrix[0][2]**2 + x_matrix[1][2]**2 + x_matrix[2][2]**2 + x_matrix[3][2]**2)/4
a12 = a21 = (x_matrix[0][0]*x_matrix[0][1] + x_matrix[1][0]*x_matrix[1][1] +
             x_matrix[2][0]*x_matrix[2][1] + x_matrix[3][0]*x_matrix[3][1])/4
a13 = a31 = (x_matrix[0][0]*x_matrix[0][2] + x_matrix[1][0]*x_matrix[1][2] +
             x_matrix[2][0]*x_matrix[2][2] + x_matrix[3][0]*x_matrix[3][2])/4
a23 = a32 = (x_matrix[0][1]*x_matrix[0][2] + x_matrix[1][1]*x_matrix[1][2] +
             x_matrix[2][1]*x_matrix[2][2] + x_matrix[3][1]*x_matrix[3][2])/4

denominator = np.linalg.det([
    [1, mx1, mx2, mx3],
    [mx1, a11, a12, a13],
    [mx2, a12, a22, a32],
    [mx3, a13, a23, a33]
])

numerator_b0 = np.linalg.det([
    [my, mx1, mx2, mx3],
    [a1, a11, a12, a13],
    [a2, a12, a22, a32],
    [a3, a13, a23, a33]
])

numerator_b1 = np.linalg.det([
    [1, my, mx2, mx3],
    [mx1, a1, a12, a13],
    [mx2, a2, a22, a32],
    [mx3, a3, a23, a33]
])

numerator_b2 = np.linalg.det([
    [1, mx1, my, mx3],
    [mx1, a11, a1, a13],
    [mx2, a12, a2, a32],
    [mx3, a13, a3, a33]
])

numerator_b3 = np.linalg.det([
    [1, mx1, mx2, my],
    [mx1, a11, a12, a1],
    [mx2, a12, a22, a2],
    [mx3, a13, a23, a3]
])

b0 = numerator_b0/denominator
b1 = numerator_b1/denominator
b2 = numerator_b2/denominator
b3 = numerator_b3/denominator

print("b0:", "%.2f" % b0, " b1:", "%.2f" % b1, " b2:", "%.2f" % b2, " b3:", "%.2f" % b3)
print(f"Рівняння регресії: y = {b0:.2f}{b1:+.2f}*x1{b2:+.2f}*x2{b3:+.2f}*x3")
if (b0 + b1*x_matrix[0][0] + b2*x_matrix[0][1] + b3*x_matrix[0][2]) == my1:
    print("b0 + b1*X11 + b2*X12 + b3*X13=my1")

print("b0 + b1*X11 + b2*X12 + b3*X13 =", "%.2f" % (b0 + b1*x_matrix[0][0] + b2*x_matrix[0][1] + b3*x_matrix[0][2]),
      "| my1 =", "%.2f" % my1)
print("b0 + b1*X21 + b2*X22 + b3*X23 =", "%.2f" % (b0 + b1*x_matrix[1][0] + b2*x_matrix[1][1] + b3*x_matrix[1][2]),
      "| my2 =", "%.2f" % my2)
print("b0 + b1*X31 + b2*X32 + b3*X33 =", "%.2f" % (b0 + b1*x_matrix[2][0] + b2*x_matrix[2][1] + b3*x_matrix[2][2]),
      "| my3 =", "%.2f" % my3)
print("b0 + b1*X41 + b2*X42 + b3*X43 =", "%.2f" % (b0 + b1*x_matrix[3][0] + b2*x_matrix[3][1] + b3*x_matrix[3][2]),
      "| my4 =", "%.2f" % my4)

x_matrix_normal = [
    [1, -1, -1, -1],
    [1, -1, 1, 1],
    [1, 1, -1, 1],
    [1, 1, 1, -1]
]

# find dispersion
S2 = []
for i in range(len(y_list)):
    S2.append(((y_list[i][0]-my_list[i])**2 + (y_list[i][1]-my_list[i])**2 + (y_list[i][2]-my_list[i])**2)/3)

S2y1 = S2[0]
S2y2 = S2[1]
S2y3 = S2[2]
S2y4 = S2[3]

"""KOHREN"""
Gp = max(S2)/sum(S2)

m = len(y_list[0])
f1 = m-1
f2 = N = len(x_matrix)
q = 0.05
# для q = 0.05, f1 = 2, f2 = 4, Gt = 0.7679
Gt = 0.7679

if Gp < Gt:
    print("Дисперсія однорідна")
else:
    print("Дисперсія не однорідна")

"""STUDENT"""
S2B = sum(S2)/N
S2beta = S2B/(N*m)
Sbeta = np.sqrt(S2beta)

beta0 = (my1*x_matrix_normal[0][0] + my2*x_matrix_normal[1][0] + my3*x_matrix_normal[2][0] +
         my4*x_matrix_normal[3][0])/4
beta1 = (my1*x_matrix_normal[0][1] + my2*x_matrix_normal[1][1] + my3*x_matrix_normal[2][1] +
         my4*x_matrix_normal[3][1])/4
beta2 = (my1*x_matrix_normal[0][2] + my2*x_matrix_normal[1][2] + my3*x_matrix_normal[2][2] +
         my4*x_matrix_normal[3][2])/4
beta3 = (my1*x_matrix_normal[0][3] + my2*x_matrix_normal[1][3] + my3*x_matrix_normal[2][3] +
         my4*x_matrix_normal[3][3])/4

t0 = abs(beta0)/Sbeta
t1 = abs(beta1)/Sbeta
t2 = abs(beta2)/Sbeta
t3 = abs(beta3)/Sbeta

f3 = f1*f2
# t_tab = 2.306  # для значення f3 = 8, t табличне = 2,306
# print("T_tab:", t_tab)
t_tab = scipy.stats.t.ppf((1 + (1-q))/2, f3)
print("t табличное:", t_tab)
if t0 < t_tab:
    b0 = 0
    print("t0<t_tab; b0=0")
if t1 < t_tab:
    b1 = 0
    print("t1<t_tab; b1=0")
if t2 < t_tab:
    b2 = 0
    print("t2<t_tab; b2=0")
if t3 < t_tab:
    b3 = 0
    print("t3<t_tab; b3=0")

y1_hat = b0 + b1*x_matrix[0][0] + b2*x_matrix[0][1] + b3*x_matrix[0][2]
y2_hat = b0 + b1*x_matrix[1][0] + b2*x_matrix[1][1] + b3*x_matrix[1][2]
y3_hat = b0 + b1*x_matrix[2][0] + b2*x_matrix[2][1] + b3*x_matrix[2][2]
y4_hat = b0 + b1*x_matrix[3][0] + b2*x_matrix[3][1] + b3*x_matrix[3][2]

print(f"y1_hat = {b0:.2f}{b1:+.2f}*x11{b2:+.2f}*x12{b3:+.2f}*x13 "
      f"= {y1_hat:.2f}")
print(f"y2_hat = {b0:.2f}{b1:+.2f}*x21{b2:+.2f}*x22{b3:+.2f}*x23"
      f" = {y2_hat:.2f}")
print(f"y3_hat = {b0:.2f}{b1:+.2f}*x31{b2:+.2f}*x32{b3:+.2f}*x33 "
      f"= {y3_hat:.2f}")
print(f"y4_hat = {b0:.2f}{b1:+.2f}*x41{b2:+.2f}*x42{b3:+.2f}*x43"
      f" = {y4_hat:.2f}")

"""FISHER"""

d = 2
f4 = N - d

S2_ad = (m/(N-d))*((y1_hat-my1)**2 + (y2_hat-my2)**2 + (y3_hat-my3)**2 + (y4_hat-my4)**2)
Fp = S2_ad/S2B
# Ft = 4.5  # для f3=8; f4=2
Ft = scipy.stats.f.ppf(1-q, f4, f3)
print("Fp:", Fp)
print("Ft:", Ft)
if Fp > Ft:
    print("Рівняння регресії не адекватно оригіналу при рівні значимості 0,05")
else:
    print("Рівняння регресії адекватно оригіналу при рівні значимості 0,05")