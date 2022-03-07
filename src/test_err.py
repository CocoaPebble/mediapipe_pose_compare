import numpy as np
from matplotlib import pyplot as plt

from calculation.calculate_angle_err import angle_err, convert_err_to_plot, each_angle_err


def read_angles(filename):
    angle_num = 8
    angles = []
    fin = open(filename, 'r')

    while True:
        line = fin.readline()
        if line == '':
            break

        line = line.split()
        line = [float(s) for s in line]
        angles.append(line)

    return angles


def draw_error_line(errors):
    x = range(0, len(errors))
    plt.plot(x, errors)
    plt.xlabel("frames")
    plt.ylabel('abs error')
    plt.show()


gt_file = 'angle_output\yogafrontmirror100_angle_output.dat'
mp_file = 'angle_output\yoga_front_test_1.dat'
gt_angles = read_angles(gt_file)
mp_anlges = read_angles(mp_file)

# err_average = []
# for i in range(200):
#     errors = angle_err(gt_angles, mp_anlges, i)
#     minerr = min(errors)
#     print('frame', i+1, minerr, np.where(errors == minerr))
#     err_average.append(minerr)
# # draw_error_line(errors)

# ave = np.average(err_average)
# print('average', ave)

keypose = 30

errors = angle_err(gt_angles, mp_anlges, keypose)
print('keypose current is', keypose)
minerr = min(errors)
min_idx = np.where(errors == minerr)
min_pos = min_idx[0][0]
print(minerr, min_pos)
# draw_error_line(errors)

start = min_pos-10
end = min_pos+10
result = each_angle_err(gt_angles, mp_anlges, keypose=keypose,
                        actual_min_pos=30, start=20, end=40)
plot_arr = convert_err_to_plot(result, 20)
print(plot_arr)
