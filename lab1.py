import random
import numpy as np

a = [2, 4, 5, 1]
X1 = [random.randrange(1, 21, 1) for i in range(8)]
X2 = [random.randrange(1, 21, 1) for i in range(8)]
X3 = [random.randrange(1, 21, 1) for i in range(8)]
y = [a[0] + a[1] * X1[i] + a[2] * X2[i] + a[3] * X3[i] for i in range(8)]


def print_x(arr):
    for i in arr:print(i, end=" ")
print("\nЗначення факторів у точках експерименту")
print("________X1_______")
print_x(X1)
print("\n________X2_______")
print_x(X2)
print("\n________X3_______")
print_x(X3)
print("\n________Y_______ ")
print_x(y)



average_y = sum(y)/len(y)
x_0 = lambda x: (max(x) + min(x)) / 2
dx = lambda x: x_0(x) - min(x)
xn = lambda x: (i - x_0(x)) / dx(x)
y_etalon = a[0] + a[1] * x_0(X1) + a[2] * x_0(X2) + a[3] * x_0(X3)

y1 = np.array(y)
y1 = y1 - y_etalon
y1 = (y1 - y_etalon) ** (2)
tochka_plany_y = max([i for i in y1])

print("\nТочка плану = ", tochka_plany_y)
print("\nСереднє значення у = ", average_y)
print("\nНормування факторів:")
print(f"\nX(0)1 = {x_0(X1)} "
      f"\ndx1 = {dx(X1)}"
      f"\nX(0)2 = {x_0(X2)}"
      f"\ndx2 = {dx(X2)}"
      f"\nX(0)3 = {x_0(X3)}"
      f"\ndx3 = {dx(X3)}"
      f"\n")
print(f"Y(et) = {y_etalon}\n")

xn1 = []
xn2 = []
xn3 = []

for i in X1: xn1.append(xn(X1))
for i in X2: xn2.append(xn(X2))
for i in X3: xn3.append(xn(X3))
print("Значення віднормованих факторів")
print("xn1")
for i in xn1: print(f"{i:.3f}", end=" ")
print("\nxn2")
for i in xn2: print(f"{i:.3f}", end=" ")
print("\nxn3")
for i in xn3: print(f"{i:.3f}", end=" ")
print("\n")

y_norm = [a[0] + a[1] * xn1[i] + a[2] * xn2[i] + a[3] * xn3[i] for i in range(8)]
print("Функція відгуку віднормованих факторів:")
for i in y_norm: print(f"{i:.3f}", end=" ")
