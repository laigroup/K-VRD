from visual_relationship_dataset import *
import numpy as np


def judgenext(pair):
    judge = False

    for aa in pair[101:105]:
        for bb in pair[206:210]:
            if abs(aa - bb) < 0.00001:
                judge = True

    return judge


def judgeRCC(s_o_pair, data, name):
    # pairs_of_data = np.array([np.concatenate((data[s_o_p[0]][1:], data[s_o_p[1]][1:],
    #                                           computing_extended_features(data[s_o_p[0]], data[s_o_p[1]])))
    #                           for s_o_p in s_o_pair])

    count = 0
    with open(r'RCC_label_' + name + '.csv', mode='w', newline='',
              encoding='utf8') as cf:
        wf = csv.writer(cf)
        # wf.writerow(
        #     ['label', 'B1xl', 'B1yl', 'B1xr', 'B1yr', 'B2xl', 'B2yl', 'B2xr', 'B2yr', 'ir(b1,b2)', 'ir(b2,b1)',
        #      'area(b1,b2)', 'area(b2,b1)', 'equlid_dist(b1,b2)', 'sin(b1,b2)', 'cos(b1,b2)'])

        for s_o_p in s_o_pair:
            if len(s_o_p) < 5:
                pair = np.array(np.concatenate((data[s_o_p[0]][1:], data[s_o_p[1]][1:],
                                                      computing_extended_features(data[s_o_p[0]], data[s_o_p[1]]))))
            else :
                pair = s_o_p
            label = '0'
            if pair[-6] == 1.0 and pair[-7] == 1.0:
                label = 'EQ'

            elif 1.0 > pair[-6] > 0 and 1.0 > pair[-7] > 0:
                label = 'PO'

            elif pair[-6] < 1.0 and pair[-7] == 1.0 and judgenext(pair) and pair[-6] > 0:
                label = 'TPP'

            elif pair[-6] == 1.0 and pair[-7] < 1.0 and judgenext(pair):
                label = 'TPP-1'

            elif pair[-6] < 1.0 and pair[-7] == 1.0 and judgenext(pair) == False and pair[-6] > 0:
                label = 'NTPP'

            elif pair[-6] == 1.0 and pair[-7] < 1.0 and judgenext(pair) == False:
                label = 'NTPP-1'

            elif pair[-6] == 0 and pair[-7] == 0 and judgenext(pair) == False:
                label = 'DC'

            elif pair[-6] == 0 and pair[-7] == 0 and judgenext(pair):
                label = 'EC'

            wf.writerow([label,s_o_p[0],s_o_p[1],s_o_p[2]])
        cf.close()

        # if label == 'EC':
        #     print(count)
        #     print(pair[101:105])
        #     print(pair[206:210])
        #     print(pair[210:212])
        #     print(pair[-7],pair[-6])
        #     print(label)
        #     print("=========================================================")

def judgercc(pair):
    if pair[-6] == 1.0 and pair[-7] == 1.0:
        label = 'EQ'
        loc = 0

    elif 1.0 > pair[-6] > 0 and 1.0 > pair[-7] > 0:
        label = 'PO'
        loc = 1

    elif pair[-6] < 1.0 and pair[-7] == 1.0 and judgenext(pair) and pair[-6] > 0:
        label = 'TPP'
        loc = 2

    elif pair[-6] == 1.0 and pair[-7] < 1.0 and judgenext(pair):
        label = 'TPP-1'
        loc = 3

    elif pair[-6] < 1.0 and pair[-7] == 1.0 and judgenext(pair) == False and pair[-6] > 0:
        label = 'NTPP'
        loc = 4
    elif pair[-6] == 1.0 and pair[-7] < 1.0 and judgenext(pair) == False:
        label = 'NTPP-1'
        loc = 5
    elif pair[-6] == 0 and pair[-7] == 0 and judgenext(pair) == False:
        label = 'DC'
        loc = 6
    elif pair[-6] == 0 and pair[-7] == 0 and judgenext(pair):
        label = 'EC'
        loc = 7
    return label, loc
