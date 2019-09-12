
import pandas as pd


def get_index_df(filename=None):
    df = pd.read_csv(filename, sep=':', delimiter=':', names=['index', 'value'])
    df['value'] = df['value'].astype(float)
    df_list = [df[df['index'] == e].reset_index(drop=True).drop(columns=['index']).rename(columns={'value': e})
               for e in df['index'].unique()]
    total_df = pd.concat(df_list, axis=1)
    return total_df


tfpack_df = get_index_df('tfpack.log')
vote_df = get_index_df('votelog.log')

# box_loss:0.598721
# center_loss:0.074337
# heading_cls_loss:2.157224
# heading_reg_loss:0.158083
# loss:13.675979
# neg_ratio:0.806152
# obj_acc:0.865007
# objectness_loss:0.093747
# pos_ratio:0.064014
# sem_cls_loss:1.135216
# size_cls_loss:1.140392
# size_reg_loss:0.036540
# vote_loss:0.608481

# box_loss:0.71504
# center_left_loss:0.080551
# center_loss:0.10991
# center_right_loss:0.029361
# heading_cls_loss:2.52
# heading_residual_loss:0.21745
# objectness_loss:0.11209
# sem_cls_loss:1.0154
# size_cls_loss:1.0852
# size_residual_loss:0.027155
# total_loss:0.98075
# vote_loss:0.10812

import matplotlib.pyplot as plt
plt.title('Result Analysis')
print(tfpack_df['total_loss'])
print(vote_df['loss'])
plt.plot(tfpack_df.index, tfpack_df['vote_loss'], label='me')
plt.plot(vote_df.index, vote_df['vote_loss'] / 10, label='official')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('index')
plt.show()