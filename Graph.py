import plotly.express as px
import os
import pandas as pd
import numpy as np

def maximize(list1):
    list2 = []
    max = 0
    for i in range(len(list1)):
        cur = list1[i]
        if (cur > max):
            max = cur

        list2.append(max)

    return list2

def main(args):
    folderpath = args[0]
    filename = args[1]
    customExt = args[2]
    defExt = args[3]
    dur = args[4]
    list1 = []
    list2 = []
    for i in range(dur):
        file = folderpath + filename + customExt + str(i) + ".txt"
        file2 = folderpath + filename + defExt + str(i) + ".txt"
        f = open(file)
        f2 = open(file2)
        data = float(f.read())
        list1.append(data)
        list1 = maximize(list1)

        data2 = float(f2.read())
        list2.append(data2)
        list2 = maximize(list2)

    df = pd.DataFrame({'Time (minutes)' : range(0, dur),
         'Acc' : np.array(list1)})

    df['HP'] = 'Custom'

    df2 = pd.DataFrame({'Time (minutes)' : range(0, dur),
         'Acc' : np.array(list2)})

    df2['HP'] = 'Default'

    df = df.append(df2)

    print(df)

    fig = px.line(df, x='Time (minutes)', y='Acc', color='HP', title="Comparison of Test Accuracy for Credit Approval")
    fig.show()

    return fig


main(["./Temp/TestingAcc/", "parallel_Kaggle-data-malware-new", "__test_acc", "_true_def_test_acc", 60])