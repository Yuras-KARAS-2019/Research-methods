import random
import time

start_time = time.time()

LIMIT = 20

a0, a1, a2, a3 = [random.randint(1, 100) for i in range(4)]      # initialization of variables

planning_matrix = [[random.randint(0, LIMIT),
                    random.randint(0, LIMIT),
                    random.randint(0, LIMIT)] for i in range(8)]  # create matrix

print("----------- three-factor experiment ------------")
list_y = []  # list for the value of feedback functions
for i in range(len(planning_matrix)):
    print(planning_matrix[i])
    list_y.append(a0 +
                  a1 * planning_matrix[i][0] +
                  a2 * planning_matrix[i][1] +
                  a3 * planning_matrix[i][2])  # calculate every element for list_y

list_for_x1 = []
list_for_x2 = []
list_for_x3 = []

for i in range(len(planning_matrix)):
    list_for_x1.append(planning_matrix[i][0])
    list_for_x2.append(planning_matrix[i][1])
    list_for_x3.append(planning_matrix[i][2])


x1_medium = (min(list_for_x1) + max(list_for_x1)) / 2
x2_medium = (min(list_for_x2) + max(list_for_x2)) / 2
x3_medium = (min(list_for_x3) + max(list_for_x3)) / 2
print("---------------- medium values -----------------")
print("medium x1: {0}, medium x2: {1}, medium x3: {2}".format(x1_medium, x2_medium, x3_medium))

dx_1 = x1_medium - min(list_for_x1)
dx_2 = x2_medium - min(list_for_x2)
dx_3 = x3_medium - min(list_for_x3)

Y_standard = a0 + a1 * x1_medium + a2 * x2_medium + a3 * x1_medium

list_x1_normalized = [(i - x1_medium) / dx_1 for i in list_for_x1]
list_x2_normalized = [(i - x2_medium) / dx_2 for i in list_for_x2]
list_x3_normalized = [(i - x3_medium) / dx_3 for i in list_for_x3]
print("------------------ normalized ------------------")
for i in range(len(list_x1_normalized)):
    print("{0:+4.3f}".format(list_x1_normalized[i]), end=" ")
    print("{0:+4.3f}".format(list_x2_normalized[i]), end=" ")
    print("{0:+4.3f}".format(list_x3_normalized[i]))

result_y = []
for i in list_y:
    result_y.append((i - Y_standard) ** 2)
# Show result:
print("-------------------- result --------------------")
print(planning_matrix[result_y.index(max(result_y))], ": " + "Y = {0}".format(list_y[result_y.index(max(result_y))]))
print("Y standard =", Y_standard)
print("List y :", list_y)
print("List with criterion:", result_y)
print("Program was executed in {0} seconds".format(round((time.time() - start_time), 5)))
