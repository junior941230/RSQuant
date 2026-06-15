import math
x = -0.10
y = -0.10
x = 0.41
y = 0.41
L = 90
cosval = (x**2 + y**2 - L**2 - L**2) / (2 * L * L)
if cosval > 1:
    cosval = 1
if cosval < -1:
    cosval = -1
theta2 = math.acos(cosval)
alpha = math.atan2(y, x)
beta = math.atan2(L * math.sin(theta2), L + L * math.cos(theta2))
theta1 = alpha - beta
print("theta1:", math.degrees(theta1))
print("theta2:", math.degrees(theta2 + theta1))

theta2 = -math.acos(cosval)
alpha = math.atan2(y, x)
beta = math.atan2(L * math.sin(theta2), L + L * math.cos(theta2))
theta1 = alpha - beta
print("theta1:", math.degrees(theta1))
print("theta2:", math.degrees(theta2 + theta1))

# theta1 = math.radians(-404.35)
# theta2 = math.radians(134.35)
# x = L * math.cos(theta1) + L * math.cos(theta1 + theta2)
# y = L * math.sin(theta1) + L * math.sin(theta1 + theta2)
# print("x:", x)
# print("y:", y)
