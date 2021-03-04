import torch
import torch.nn as nn
import datetime
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from matplotlib import cm
from sklearn.model_selection import train_test_split
import load_data
import functions

np.random.seed(100)


wl1 = 50
wl2 = 100

df_dx = pd.DataFrame()

all_files = [load_data.file14, load_data.file15, load_data.file16, load_data.file17, load_data.file19, load_data.file20, load_data.file21, load_data.file22, load_data.file23]

for i,j in enumerate([14,15,16,17,19,20,21,22,23]):
    df_dx['x%(j)s' % {"j": j}]  =  all_files[i]['x_shifted']
    df_dx['y%(j)s' % {"j": j}]  =  functions.smooth(all_files[i]['y_norm'], wl1)
    df_dx['derivative y%(j)s' %{"j": j}] = np.divide(df_dx['y%(j)s'% {"j": j}].diff(), \
                                                     df_dx['x%(j)s'% {"j": j}].diff())
    df_dx['smooth derivative y%(j)s' %{"j": j}] = functions.smooth(df_dx['derivative y%(j)s' %{"j": j}], wl2)


li = []

for i in ([14, 15, 16, 19, 20, 21]):
    crossing = df_dx[(df_dx['derivative y%(i)s' %{"i": i}] > 0) & (df_dx['x%(i)s'% {"i": i}] > 1)]['x%(i)s'% {"i": i}].min()
    li.append(crossing)

dE1 = min(li)
dE1_min = np.round(dE1, 3)
print("The value of dE1 is", dE1_min)

nrows, ncols = 3,1
gs = matplotlib.gridspec.GridSpec(nrows,ncols)
plt.figure(figsize=(ncols*7,nrows*4.5))

cm_subsection = np.linspace(0,1,24)
colors = [cm.viridis(x) for x in cm_subsection]

hfont = rc('font',**{'family':'sans-serif','sans-serif':['Sans Serif']})

for i in range(2):
    ax = plt.subplot(gs[i])
    ax.set_xlim([0,9])
    ax.tick_params(which='major',direction='in',length=7)
    ax.tick_params(which='minor',length=8)
    plt.axhline(y=0, color='black', linewidth=1, alpha=.8)
    plt.axvline(x=0, color='darkgray', linestyle='--', linewidth = 1)
    #plt.axvline(x=dE1, color='darkgray', linestyle='--', linewidth = 1, label='$\Delta$E1' %{'s': dE1})

    for j in ([17,22,23]):
        if i == 0:
            p2 = ax.plot(df_dx['x%(i)s'% {"i": j}],df_dx['derivative y%(i)s' %{"i": j}], color=colors[j], label='%(i)s' %{"i": j})

    for j in ([14,15,16,19,20,21]):
        k = j-3

        if i == 0:
            p1 = ax.plot(df_dx['x%(i)s'% {"i": j}],df_dx['derivative y%(i)s' %{"i": j}], color=colors[-k], label='%(i)s' %{"i": j})
            ax.set_ylim([-.002, .001])
            ax.set_xlim([0, 6])
            ax.set_ylabel('dI/dE',fontsize=18)
            ax.set_yticks([-0.002, -0.001, 0, 0.001])
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.legend(loc=2, fontsize=16)

    for j in ([17,22,23]):
        if i == 1:
            ax.axhline(y=1, linestyle='-', color='gray')
            p1 = ax.plot(df_dx['x%(i)s'% {"i": j}], \
                         np.divide(df_dx['derivative y14'],df_dx['derivative y%(i)s'%{"i": j}]), 'k--', label='Ratio sp4/sp'%{"i":j})

            ax.axvline(x=1.65, linestyle='--')
            ax.set_ylim([-1, 2])
            ax.set_xlim([.5,3.5])
            ax.set_ylabel('R = dI/dE(sample) / dI/dE(vac)', fontsize=18)
            ax.set_xlabel('$\Delta$E (eV)', fontsize=218)
            ax.legend()

    if i == 0:
        ax.tick_params(labelbottom=True)
        ax.tick_params(which='major', length= 10, labelsize=18)
        ax.tick_params(which='minor', length= 10, labelsize=10)
    if i == 1:
        ax.set_xlabel('Energy loss (eV)', fontsize=24)
        ax.tick_params(length= 10, labelsize=18)
        ax.tick_params(which='major', length= 10, labelsize=18)
        ax.tick_params(which='minor', length= 10, labelsize=10)

plt.tight_layout()
#plt.savefig("Derivatives.pdf")
plt.show()

df_vacmean = pd.DataFrame()
nbins = 150
df_vacuum = load_data.df_vacuum[(load_data.df_vacuum['x_shifted'] < 20) & (load_data.df_vacuum['x_shifted'] > -.5)]
df_vacmean['x'] = np.linspace(df_vacuum['x_shifted'].min(),df_vacuum['x_shifted'].max(), nbins)
df_vacmean['y'], df_vacmean['sigma'] = functions.binned_statistics(df_vacuum['x_shifted'], (df_vacuum['y']), nbins)[0:2]
df_vacmean['ratio'] = np.divide(df_vacmean['y'], df_vacmean['sigma'])

dE2 = df_vacmean['x'][df_vacmean['ratio'] < 1].min()
dE2 = np.round(dE2)
print("The value for dE_II is", (dE2))

nrows, ncols = 1,1
gs = matplotlib.gridspec.GridSpec(nrows,ncols)
plt.figure(figsize=(ncols*5,nrows*3.5))

cm_subsection = np.linspace(0,1,24)
colors = [cm.viridis(x) for x in cm_subsection]

hfont = rc('font',**{'family':'sans-serif','sans-serif':['Sans Serif']})

ax = plt.subplot(gs[0])
ax.set_title('Intensity to sigma ratio', fontsize=16)
ax.set_xlim([-1,15])
ax.set_xlabel('Energy loss (eV)', fontsize=14)
ax.tick_params(which='major',direction='in',length=7, labelsize=14)
ax.tick_params(which='minor',length=8)
p1 = ax.plot(df_vacmean['x'],functions.smooth(np.divide(df_vacmean['y'], df_vacmean['sigma']), 10), color=colors[0])
ax.axhline(y=1, linestyle='-')
ax.axvline(x=dE2, linestyle='dotted', linewidth='2')
plt.show()

print('The values of dE1 and dE2:', np.round(dE1,2), "eV and", dE2, "eV")


df_window = load_data.df[(load_data.df['x_shifted'] < dE1) & (load_data.df['x_shifted'] > -.5)]
df_window_vacuum = load_data.df_vacuum[(load_data.df_vacuum['x_shifted'] <= dE1) & (load_data.df_vacuum['x_shifted'] > -.5)]

df_mean, df_vacmean = pd.DataFrame(), pd.DataFrame()
nbins = 30

test1 = df_window['x_shifted'].min()
test2 = df_window['x_shifted'].max()

df_mean['x'] = np.linspace(df_window['x_shifted'].min(),df_window['x_shifted'].max(), nbins)
df_mean['y'], df_mean['sigma'] = functions.binned_statistics(df_window['x_shifted'], np.log(df_window['y']), nbins)[0:2]

df_vacmean['x'] = np.linspace(df_window_vacuum['x_shifted'].min(),df_window_vacuum['x_shifted'].max(), nbins)
df_vacmean['y'], df_vacmean['sigma'] = functions.binned_statistics(df_window_vacuum['x_shifted'], np.log(df_window_vacuum['y']), nbins)[0:2]

print("Training data points for DeltaE > DeltaE_I have been removed.")
print("Experimental mean and sigma are calculated.")

min_x = dE2
max_x = 16
N_pseudo = 20

df_pseudo = pd.DataFrame({'x':np.linspace(min_x, max_x, N_pseudo),'y': .5 * np.ones(N_pseudo), 'sigma': .08 * np.ones(N_pseudo)})
df_full = pd.concat([df_mean, df_pseudo])

print('Pseudo data points added for Delta E > DeltaE_II')
print('Training data set "df_full" has been created')

df_full.describe()

df_train_full = df_full
df_train_full = df_train_full.drop_duplicates(subset = ['x']) # Only keep one copy per x-value

N_full = len(df_train_full['x'])

full_x = np.copy(df_train_full['x']).reshape(N_full,1)
full_y = np.copy(df_train_full['y']).reshape(N_full,1)
full_sigma = np.copy(df_train_full['sigma']).reshape(N_full,1)

N_pred = 3000
pred_min = -.5
pred_max = 20
predict_x = np.linspace(pred_min,pred_max,N_pred).reshape(N_pred,1)

plt.figure(figsize=(10,5))
plt.plot(full_x, full_y, '.', label='train')

plt.axvline(x=dE1, color='lightgray')
plt.axvline(x=dE2, color='lightgray')
plt.title('Visualization of training data', fontsize=15)
plt.ylabel('Log intensity', fontsize=14)
plt.xlabel('Energy loss (eV)', fontsize=14)
plt.legend(fontsize=14)
plt.show()

Nrep = 500

full_y_reps = np.zeros(shape=(N_full, Nrep))
i=0
while i < Nrep:
        full_rep = np.random.normal(0, full_sigma)
        full_y_reps[:,i] = (full_y + full_rep).reshape(N_full)
        i+=1

std_reps = np.std(full_y_reps, axis=1)
mean_reps = np.mean(full_y_reps, axis=1)

print('MC pseudo data has been created for %(nrep)s replicas' %{"nrep": Nrep})


N_train = int(.8 * N_full)
N_test = int(.2 * N_full)


############################

class MLP(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        # Initialize the modules we need to build the network
        self.linear1 = nn.Linear(num_inputs, 10)
        self.linear2 = nn.Linear(10, 15)
        self.linear3 = nn.Linear(15, 5)
        self.output = nn.Linear(5, num_outputs)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.output(x)
        return x




def loss_fn(input, target, error):
    loss = torch.mean(((input - target)/error) ** 2)
    print(loss)
    return loss

# TODO: write a dataloader
def get_batch(rep_i):
    pass
    # return x, y


def training_loop(n_epochs):
    n_rep = 100
    display_step = 1000
    for i in range(n_rep):
        print("Started training on replica number {}".format(i))
        model = MLP(num_inputs=1, num_outputs=1)
        optimizer = optim.Adam(model.parameters(), lr=6 * 1e-3)
        # TODO: rewrite to include pytorch directly, see pyfiles/train_nn.py
        full_y = full_y_reps[:, i].reshape(N_full, 1)
        train_x, test_x, train_y, test_y, train_sigma, test_sigma = train_test_split(full_x, full_y, full_sigma, test_size=.2)
        train_x, test_x = train_x.reshape(N_train, 1), test_x.reshape(N_test, 1)
        train_y, test_y = train_y.reshape(N_train, 1), test_y.reshape(N_test, 1)
        train_sigma, test_sigma = train_sigma.reshape(N_train, 1), test_sigma.reshape(N_test, 1)

        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)
        train_sigma = train_x = torch.from_numpy(train_sigma)
        # train_data_x, train_data_y, train_errors = get_batch(i)
        for epoch in range(1, n_epochs + 1):
            output = model(train_x.float())
            loss_train = loss_fn(output, train_y, train_sigma)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            if epoch % display_step == 0:
                print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train))


# model = MLP(num_inputs=1, num_outputs=1)
# optimizer = optim.Adam(model.parameters(), lr=6*1e-3)

training_loop(
    n_epochs=25000
)