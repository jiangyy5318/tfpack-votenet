
import pandas as pd
df = pd.read_csv('hehe.txt', sep=':', delimiter=':', names=['index', 'value'])

df_list = [df[df['index'] == e].reset_index(drop=True).drop(columns=['index']).rename(columns={'value': e})
           for e in df['index'].unique()]
total_df = pd.concat(df_list, axis=1)

print(total_df)
import matplotlib.pyplot as plt
plt.title('Result Analysis')
for e in total_df.columns:
    plt.plot(total_df.index,total_df[e], label=e)
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('index')
plt.show()