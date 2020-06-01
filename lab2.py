
import math
import random

x1min = 15
x1max = -5
x2min = -35
x2max = 10
ymax = 230
ymin = 130
m = 5

x1l = [-1, 1, -1]
x2l = [-1, -1, 1]

y1l = []
y2l = []
y3l = []

for i in range(5):
    y1l.append(random.randint(ymin, ymax))
    y2l.append(random.randint(ymin, ymax))
    y3l.append(random.randint(ymin, ymax))

y1 = 0
y2 = 0
y3 = 0

for i in range(5):
    y1 += y1l[i]
    y2 += y2l[i]
    y3 += y3l[i]

y1 /= 5
y2 /= 5
y3 /= 5

sigma1 = 0
sigma2 = 0
sigma3 = 0

for i in range(5):
    sigma1 += math.pow((y1l[i] - y1), 2)
    sigma2 += math.pow((y2l[i] - y2), 2)
    sigma3 += math.pow((y3l[i] - y3), 2)

sigma1 /= m
sigma2 /= m
sigma3 /= m

sigma0 = math.sqrt((2 * (2 * m - 2)) / (m * (m - 4)))

if sigma1 > sigma2:
    Fuv1 = sigma1 / sigma2

else:
    Fuv1 = sigma2 / sigma1

if sigma3 > sigma1:
    Fuv2 = sigma3 / sigma1

else:
    Fuv2 = sigma1 / sigma3

if sigma3 > sigma2:
    Fuv3 = sigma3 / sigma2

else:
    Fuv3 = sigma2 / sigma3

Ouv1 = ((m - 2) / m) * Fuv1
Ouv2 = ((m - 2) / m) * Fuv2
Ouv3 = ((m - 2) / m) * Fuv3

Ruv1 = math.fabs(Ouv1 - 1) / sigma0
Ruv2 = math.fabs(Ouv2 - 1) / sigma0
Ruv3 = math.fabs(Ouv3 - 1) / sigma0

p = [1.73, 2.16, 2.43, 2.62, 2.75, 2.9, 3.08]

if m <= 4:
    Rk = p[0]

elif m > 4 and m <= 6:
    Rk = p[1]

elif m > 6 and m <= 8:
    Rk = p[2]

elif m > 8 and m <= 10:
    Rk = p[3]

elif m > 10 and m <= 12:
    Rk = p[4]

elif m > 12 and m <= 15:
    Rk = p[5]
else:
    Rk = p[6]

if Ruv1 < Rk and Ruv2 < Rk and Ruv3 < Rk:
    print("Дисперсія однорідна")

mx1 = 0
mx2 = 0
a1 = 0
a2 = 0
a3 = 0

for i in range(3):
    mx1 += x1l[i]
    mx2 += x2l[i]
    a1 += math.pow(x1l[i], 2)
    a2 += x1l[i] * x2l[i]
    a3 += math.pow(x2l[i], 2)

mx1 /= 3
mx2 /= 3
my = (y1 + y2 + y3) / 3

a1 /= 3
a2 /= 3
a3 /= 3

a11 = (x1l[0] * y1 + x1l[1] * y2 + x1l[2] * y3) / 3
a22 = (x2l[0] * y1 + x2l[1] * y2 + x2l[2] * y3) / 3

b0 = (my * a1 * a3 + a11 * a2 * mx2 + mx1 * a2 * a22 - mx2 * a1 * a22 - a2 * a2 * my - a11 * mx1 * a3) / (
            a1 * a3 + a2 * mx1 * mx2 + mx1 * mx2 * a2 - mx2 * a1 * mx2 - a2 * a2 - mx1 * mx1 * a3)
b1 = (a11 * a3 + mx1 * a22 * mx2 + my * a2 * mx2 - mx2 * a11 * mx2 - mx1 * my * a3 - a22 * a2) / (
            a1 * a3 + a2 * mx1 * mx2 + mx1 * mx2 * a2 - mx2 * a1 * mx2 - a2 * a2 - mx1 * mx1 * a3)
b2 = (a1 * a22 + mx1 * a2 * my + mx1 * a11 * mx2 - my * a1 * mx2 - mx1 * mx1 * a22 - a2 * a11) / (
            a1 * a3 + a2 * mx1 * mx2 + mx1 * mx2 * a2 - mx2 * a1 * mx2 - a2 * a2 - mx1 * mx1 * a3)

print("y = " + str(b0) + " + " + str(b1) + "x1 + " + str(b2) + "x2")

dx1 = math.fabs(x1max - x1min) / 2
dx2 = math.fabs(x2max - x2min) / 2
x10 = (x1max + x1min) / 2
x20 = (x2max + x2min) / 2

a0 = b0 - b1 * x10 / dx1 - b2 * x20 / dx2
a1 = b1 / dx1
a2 = b2 / dx2

print("y = " + str(a0) + " + " + str(a1) + "x1 + " + str(a2) + "x2")
"""print(y1l)
print(y2l)
print(y3l)"""
