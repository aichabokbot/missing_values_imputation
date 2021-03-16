from src.data_manager import DataManager
import matplotlib.pyplot as plt
import numpy as np

na_props = np.linspace(0, 0.95, num=10)
nrmses = []

for na_prop in na_props:
    dm = DataManager(na_prop=0.05)
    pred, nrmse_value = dm.last_value_carried_forward()
    print(na_prop, nrmse_value)
    nrmses.append(nrmse_value)

plt.plot(na_props, nrmses)
plt.savefig('test.png')
