# Importy
import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time, os
# import sys
import random
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Dynamicky learning rate - step scheduler

# writer = SummaryWriter("runs/diplomovka")
# writer.close()

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Pouzitie GPU
    print("Pouzitie GPU.")
else:
    device = torch.device("cpu")     # Pouzitie CPU ak CUDA nie je dostupna
    print("Pouzitie CPU.")


def data_preload(start_data, end_data, reduced, priecinok_cesta=f"{os.getcwd()}\\data"):
    datasets = dict()
    # Allokuje priestor pre FRF
    frf_data_len = np.genfromtxt(f"{priecinok_cesta}/FRF_databaza/FRF_{1}.csv", delimiter=',', skip_header=0)[0,
                   :].__len__() if not reduced else np.genfromtxt(f"{priecinok_cesta}/FRF_databaza/FRF_{1}.csv", delimiter=',',
                                                                  skip_header=0)[0, ::2].__len__()
    frf_data = np.zeros([1, frf_data_len * 2])

    vuf_data = np.zeros([1, 2])
    damp_data = np.zeros([1, 2])

    # Vytvori dictionary so vsetkymi datami s priradenym indexovanim
    for i, data_index in enumerate(range(start_data, end_data + 1)):
        temp_dict = dict()
        temp_dict["data_index"] = data_index
        # Dictionary FRF
        frf_dataset = np.genfromtxt(f"{priecinok_cesta}/FRF_databaza/FRF_{data_index}.csv", delimiter=',',
                                    skip_header=0) if not reduced else np.genfromtxt(
            f"{priecinok_cesta}/FRF_databaza/FRF_{data_index}.csv", delimiter=',', skip_header=0)[:, ::2]
        frf_data[0, :frf_data_len] = frf_dataset[0, :]
        frf_data[0, frf_data_len:] = np.log(frf_dataset[1, :])  # Logaritmovanie y hodnoty
        temp_dict["frf"] = frf_data

        # Dictionary VUF
        vuf_data[0, :] = np.genfromtxt(f"{priecinok_cesta}/VUT_&_pomerne_tlmenie/vut_pomerne_tlmenie_{data_index}.csv", delimiter=',',
                                       skip_header=0)[0, :]
        temp_dict["vuf"] = vuf_data

        # Dictionary Tlmenie
        damp_data[0, :] = np.genfromtxt(f"{priecinok_cesta}/VUT_&_pomerne_tlmenie/vut_pomerne_tlmenie_{data_index}.csv", delimiter=',',
                                        skip_header=0)[1, :]
        temp_dict["tlmenie"] = damp_data

        datasets[f"{i}"] = temp_dict
    return datasets

# Nacitavanie random input datasetov
# def load_random_datasets(random_data_indexes, reduced=False ,priecinok_cesta=f"{os.getcwd()}\\data"):
#     dataset = np.genfromtxt(f"{priecinok_cesta}/FRF_databaza/FRF_{1}.csv", delimiter=',', skip_header=0) if not reduced else np.genfromtxt(f"{priecinok_cesta}/FRF_databaza/FRF_{1}.csv", delimiter=',', skip_header=0)[:, ::2]
#     data_len = dataset[0, :].__len__()
#     data_num = random_data_indexes.shape[1]
#     datasets = np.zeros([data_num, 1, data_len*2])
#     # nacitavanie_string = "Nacitavanie inputov"
#     # print(nacitavanie_string, end="")
#     for index, random_data_index in enumerate(random_data_indexes[0, :]):
#         # if index % 10 == 0:
#         #     if nacitavanie_string == "Nacitavanie inputov...":
#         #         nacitavanie_string = "Nacitavanie inputov"
#         #     else:
#         #         nacitavanie_string += "."
#         # print(f"\r{index+1}/{data_num} {nacitavanie_string}", end="")
#         dataset = np.genfromtxt(f"{priecinok_cesta}/FRF_databaza/FRF_{random_data_index}.csv", delimiter=',', skip_header=0) if not reduced else np.genfromtxt(f"{priecinok_cesta}/FRF_databaza/FRF_{random_data_index}.csv", delimiter=',', skip_header=0)[:, ::2]
#         datasets[index, 0, :data_len] = dataset[0, :]
#         datasets[index, 0, data_len:] = np.log(dataset[1, :])  # Logaritmovanie y hodnoty
#     # print(f"\r{index+1}/{data_num} Inputy nacitane")
#     return datasets
#
#
# # Nacitanie random verification datasetov
# def load_random_verification_dataset(random_data_indexes, priecinok_cesta=f"{os.getcwd()}\\data"):
#     dataset = np.genfromtxt(f"{priecinok_cesta}/VUT_&_pomerne_tlmenie/vut_pomerne_tlmenie_{1}.csv", delimiter=',', skip_header=0)
#     data_len = dataset[0, :].__len__()
#     data_num = random_data_indexes.shape[1]
#     datasets = np.zeros([data_num, 2, data_len])
#     # nacitavanie_string = "Nacitavanie outputov"
#     # print(nacitavanie_string, end="")
#     for index, random_data_index in enumerate(random_data_indexes[0, :]):
#         # if index % 10 == 0:
#         #   if nacitavanie_string == "Nacitavanie outputov...":
#         #       nacitavanie_string = "Nacitavanie outputov"
#         #   else:
#         #       nacitavanie_string += "."
#         # print(f"\r{index+1}/{data_num} {nacitavanie_string}", end="")
#         dataset = np.genfromtxt(f"{priecinok_cesta}/VUT_&_pomerne_tlmenie/vut_pomerne_tlmenie_{random_data_index}.csv", delimiter=',', skip_header=0)
#         datasets[index, 0, :] = dataset[0, :]
#         datasets[index, 1, :] = dataset[1, :]
#     # print(f"\r{index+1}/{data_num} Overovacie outputy nacitane")
#     return datasets


# Nacitanie dat na verifikaciu
# def load_verification_datasets(data_index, reduced=False, priecinok_cesta=f"{os.getcwd()}\\data"):
#     input_dataset = np.genfromtxt(f"{priecinok_cesta}/FRF_databaza/FRF_{data_index}.csv", delimiter=',', skip_header=0) if not reduced else np.genfromtxt(f"{priecinok_cesta}/FRF_databaza/FRF_{data_index}.csv", delimiter=',', skip_header=0)[:, ::2]
#     data_len = input_dataset[0, :].__len__()
#     verification_input = np.zeros([1, 1, data_len*2])
#     verification_input[0, 0, :data_len] = input_dataset[0, :]
#     verification_input[0, 0, data_len:] = np.log(input_dataset[1, :])
#     output_dataset = np.genfromtxt(f"{priecinok_cesta}/VUT_&_pomerne_tlmenie/vut_pomerne_tlmenie_{data_index}.csv", delimiter=',', skip_header=0)
#     data_len = output_dataset[0, :].__len__()
#     verification_output = np.zeros([1, 2, data_len])
#     verification_output[0, 0, :] = output_dataset[0, :]
#     verification_output[0, 1, :] = output_dataset[1, :]
#     return verification_input, verification_output


def accuracy_test(output, verification):
    output = output.cpu().detach().numpy()
    verification = verification.cpu().detach().numpy()
    out_1_acuraccy = (1 - abs(output[0, 0] - verification[0, 0]) / verification[0, 0]) * 100
    out_2_acuraccy = (1 - abs(output[0, 1] - verification[0, 1]) / verification[0, 1]) * 100
    # print("\nout_1", output[0,0], "out_2", output[0,1])
    # print("ver_1", verification[0,0], "ver_2", verification[0, 1])
    # print("1_accuracy", out_1_acuraccy)
    # print("2_accuracy", out_2_acuraccy)
    return (out_1_acuraccy+out_2_acuraccy)/2


# Vytvaranie nahodnych batch-ov datasetov
# def create_random_choice(num_of_data_in_batch, num_of_datasets):
#     mylist = np.linspace(1, num_of_datasets, num_of_datasets, dtype=int).tolist()
#     i = 0
#     num_of_batches = num_of_datasets//num_of_data_in_batch
#     random_datasets = np.zeros([num_of_batches, 1, num_of_data_in_batch], dtype=int)
#     dataset_index = 0
#     while True:
#         temp = random.choice(mylist)
#         mylist.remove(temp)
#         # print(f"{i} temp:", temp)
#         random_datasets[dataset_index, 0, i] = temp
#         i += 1
#         if i >= num_of_data_in_batch:
#             dataset_index += 1
#             i = 0
#         if dataset_index >= num_of_batches:
#             break
#     return random_datasets


# Fukcia na vypocet casu
def calc_time(time_left):
    hod = f"0{int(time_left // (60*60))}" if len(str(int(time_left // (60*60)))) < 2 else int(time_left // (60*60))
    min = f"0{int((time_left % (60*60)) // 60)}" if len(str(int(time_left % (60*60) // 60))) < 2 else int((time_left % (60*60)) // 60)
    sec = f"0{int(time_left % (60*60) % 60)}" if len(str(int(time_left % (60*60) % 60))) < 2 else int(time_left % (60*60) % 60)
    return hod, min, sec


# Definicia Modelu/Neuralnej Siete
class VUF_odhadovac(nn.Module):
    def __init__(self, n_input_layers, n_output_layers, h1, h2, h3, h4):
        super(VUF_odhadovac, self).__init__()
        # Zadefinovanie vrstiev
        self.sequential_layers = nn.Sequential(
            nn.Linear(n_input_layers, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, h4),
            nn.ReLU())
        self.output_layer = nn.Linear(h4, n_output_layers) # Output vrstva

    def forward(self, x):
        # Definicia forward propagation
        x = self.sequential_layers(x)
        x = self.output_layer(x)  # Posledna vrstva ma linernu aktivacnu funkciu
        return x

priecinok_cesta = f"{os.getcwd()}\\data"
# ----------------------------------------------------------------------------------------------------------------------
# Nacitanie dat --------------------------------------------------------------------------------------------------------
# start_data  -> cislo PRVYCH dat intervalu na ktorom sa bude trenovat
# end_data    -> cislo POSLEDNYCH dat intervalu na ktorom sa bude trenovat
# num_of_data_in_batch -> pocet nahodnych datasetov vchadzjucich do trenovacieho procesu
start_data, end_data = (1, 8000)
ver_start_data, ver_end_data = (8001, 10000)
epochs = 50                        # Pocet epoch
reduced = True                      # Zredukuje vstupne data na polovicu a tym sa zmensi aj pocet neur. v input vrstve
average_loss_accuracy_num = 10               # pocet iteracii na ulozenie loss
output_option = "vuf"   # "tlmenie"
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

print(f"Nacitane - pocet trenovacich datasetov: {end_data}\n\
         - Trenovanie:                  {output_option}\n\
         - pocet epoch:                 {epochs}\n\
         - Redukcia dat:                {'Ano' if reduced else 'Nie'}\n")

# Vyber poctu neuronov input vrstvy
example_input = data_preload(1,1,reduced)
n_input_layers = example_input["0"]["frf"].shape[1]
n_output_layers = 2

# ----------------------------------------------------------------------------------------------------------------------
# Inicializacia Modelu -------------------------------------------------------------------------------------------------
model_type = "VUF"                  # Trenovanie bude na VUF alebo pomerne tlmenie
learning_rate = 0.001              # Learning rate
h1, h2, h3, h4 = 624, 312, 156, 78          # Velkost jednotlivych vrstiev v modeli NN
layers_num = 5                      # Celkovy pocet vrstiev
# Nastavenie ukladania
model_hyperparameters = f"model_{str(learning_rate).replace('.', 'dot')}lr_{epochs}e_{end_data}d"
model_parameters = f"model_{model_type}_{layers_num}l_{n_input_layers}il_{h1}_{h2}_{h3}_{h4}_{n_output_layers}ol"
model_name = "model.pth"
model = VUF_odhadovac(n_input_layers, n_output_layers, h1, h2, h3, h4).to(device) # Inicializacia modelu
# Loss & Optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
torch.manual_seed(111)  # Zapnut ak chcem reprodukovatelne data
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
optimizer.zero_grad()
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Zobrazenie modelu v tensorboarde
# writer.add_graph(model, torch.from_numpy(example_input).type(torch.float32))
# writer.close()
# sys.exit()

# INFO
print(f"Pocet neuronov vo vstupnej vrstve:  {n_input_layers}")
print(f"Pocet neuronov vo vystupnej vrstve: {n_output_layers}")
print("\nInicializacia modelu prebehla uspesne")

# Trenovacia funkcia s random datami
total_iterations = epochs*end_data
iterations_left = epochs * end_data
loss_list = list()
accuracy_list = list()
timer_epoch = list()
hod, min, sec = "nan", "nan", "nan"
temp_loss = 0.0
temp_accuracy = 0.0
epoch_loss = 0.0
model.train()

print("Nacitavanie dat...")
preload_data = data_preload(start_data, end_data, True)
print("Data Nacitane!\nTrenovanie:\n")
shuffle_indexes = list(range(end_data-start_data))

for epoch in range(epochs):
    # Zapnutie casovaca jednej epochy
    timer_start = time.time()

    # Vytvorenie rozhadzania indexov dat
    random.shuffle(shuffle_indexes)

    # Trenovaci cyklus na rozhadzanych datach
    for i, data_index in enumerate(shuffle_indexes):
        # Nacitanie inputov a porovnavacich dat
        input_data = torch.from_numpy(preload_data[f"{data_index}"]["frf"]).type(torch.float32).to(device)
        sim_output = torch.from_numpy(preload_data[f"{data_index}"][output_option]).type(torch.float32).to(device)
        outputs = model(input_data)
        loss = criterion(outputs, sim_output)

        optimizer.zero_grad() # Vynulovanie gradientu
        loss.backward()
        optimizer.step()

        # Zapisanie do tensorboardu
        iterations_left -= 1
        # writer.add_scalar("loss", loss, total_iterations - iterations_left)

        temp_loss += loss.item()
        epoch_loss += loss.item()
        temp_accuracy += accuracy_test(outputs, sim_output)
        if (total_iterations - iterations_left) % average_loss_accuracy_num == 0:
            loss_list.append(temp_loss / average_loss_accuracy_num)
            accuracy_list.append(temp_accuracy / average_loss_accuracy_num)
            temp_loss = 0.0
            temp_accuracy = 0.0

        print(f"\rEpocha: {epoch+1}/{epochs}, Zostava: {iterations_left} iter, Odhadovany cas vypoctu: {hod}h {min}m {sec}s\t", end="")
    scheduler.step(epoch_loss / end_data)
    epoch_loss = 0.0
    # writer.close()
    timer_epoch.append(time.time() - timer_start)
    hod, min, sec = calc_time((epochs-(epoch+1))*np.mean(timer_epoch))
print("\n\nTrenovanie dokoncene!\n\n")

# Vypocet accuracy na datach na ktorych nebol model trenovany
print("Vypocet accuracy na verifikacnych datach:")
preload_verification_data = data_preload(ver_start_data, ver_end_data, reduced)
ver_data_num = ver_end_data - ver_start_data
ver_accuracy = 0.0
model.eval()
with torch.no_grad():
    for i in range(ver_data_num):
        ver_input = torch.from_numpy(preload_verification_data[f"{i}"]["frf"]).type(torch.float32).to(device)
        ver_sim_output = torch.from_numpy(preload_verification_data[f"{i}"][output_option]).type(torch.float32).to(device)
        ver_output = model(ver_input)
        ver_accuracy += accuracy_test(ver_output, ver_sim_output)

print(f"\nPresnost modelu na novych datach: {round(ver_accuracy / ver_data_num, 2)}%")
print('Oslava??')

# Create a figure and axis object
fig, ax = plt.subplots()
# Plotting the data
ax.plot(loss_list, label='Strata', color="red")
ax.set_title(f"Strata - {total_iterations}/{average_loss_accuracy_num}")
ax.set_xlabel(f"Iterácie - {total_iterations}/{average_loss_accuracy_num}")
ax.grid()
ax.legend()

# Create a figure and axis object
fig1, ax1 = plt.subplots()
# Plotting the data
ax1.plot(accuracy_list, label='Accuracy', color="green")
ax1.set_title(f"Priemerná presnosť odhadovaných dát")
ax1.set_xlabel(f"Iterácie - {total_iterations}/{average_loss_accuracy_num}")
ax1.grid()
ax1.legend()


# Ukladanie modelu a potrebnych informacii -----------------------------------------------------------------------------
if not os.path.exists(f"models/{model_parameters}"):
    os.makedirs(f"models/{model_parameters}")
if not os.path.exists(f"models/{model_parameters}/{model_hyperparameters}"):
    os.makedirs(f"models/{model_parameters}/{model_hyperparameters}")
torch.save(model, f"models/{model_parameters}/{model_hyperparameters}/{model_name}")

# Ulozenie priebehu loss
with open(f"models/{model_parameters}/{model_hyperparameters}/loss.txt", 'w') as file:
    # Write each item on a new line
    for item in loss_list:
        file.write(str(item) + '\n')

fig.savefig(f"models/{model_parameters}/{model_hyperparameters}/loss.png", format='png', dpi=300, bbox_inches='tight')
fig1.savefig(f"models/{model_parameters}/{model_hyperparameters}/accuracy.png", format='png', dpi=300, bbox_inches='tight')
# ----------------------------------------------------------------------------------------------------------------------
plt.show()
