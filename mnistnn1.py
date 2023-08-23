#naimportujeme potrebne knihovny
import numpy as np
import pathlib
import matplotlib.pyplot as plt


#nemusite mit data ulozena lokalne a muzete pro jejich nacteni pouzit napr.: Pandas, zalezi na vas
def get_dataset():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/mnist.npz") as f:
        images, labels, images_test, labels_test = f["x_train"], f["y_train"], f["x_test"], f["y_test"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    images_test = images_test.astype("float32") / 255
    images_test = np.reshape(images_test, (images_test.shape[0], images_test.shape[1] * images_test.shape[2]))
    labels_test = np.eye(10)[labels_test]

    return images, labels, images_test, labels_test

images, labels, images_test, labels_test = get_dataset()
#print(images.shape) #muzeme videt ze tento dataset ma 60 000 obrazku kazdy o 784 pixelech
#print(labels.shape) #10 je proto ze na mame jen 10 moznych cisel
#pokud tedy NN vyhodnoti vysledek jako cislo 3 bude output layer vypadat [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

number_of_images = images.shape[0]
index_array = np.arange(number_of_images)
np.random.shuffle(index_array)
images = images[index_array]
labels = labels[index_array]

#vypise se oznaceni devateho prvku
#[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#tohle skutecne je oznaceni cisla staci pouze zjistit kde je 1
#v tomhle pripade je tedy na desatem miste v nasem datasetu obrazek cisla 4
#print(labels[9])

#weights jsou na zacatku nahodne cisla
#postupnym opravovanim svych chyb si NN hodnoty prizpusoby tak aby byly presne a vedly k vysledku
#w_i_h = weights z input layer do hidden layer
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784)) #random cisla kde 784 input node jde do 20 nodes prvniho hidden layeru
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20)) #random cisla z druheho layeru do 10 outputu (mame 10 ruznych cisel)
#biases jsou taky ze zacatku nahodne a NN si je pozdeji sama trenovanim upravi
#bias je ve funkci y=ax+b zastoupeno pismenem b
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))


#preddefinujeme promenne potrebne pro nasi NN
learn_rate = 0.01
nr_correct = 0
epochs = 7 #pocet opakovani trenovaciho cyklu, projdeme vsechny obrazky petkrat

#samotna NN
for epoch in range(epochs):
    for image, label in zip(images, labels):
        #upravime image a label tak aby meli dimenzi (jinak nefunkcni)
        #print(image.shape) -> (784,)
        image.shape += (1,)
        #print(image.shape) -> (784, 1)
        label.shape += (1,)

        #kazdy pixel ma hodnoty od 0.0 do 1.0, kde bila je 1.0 a cerna je 0.0 a nahodna sediva muze byt napr.: 0.489
        #forward propagation -> hidden
        #bias + weight @ image, @ je specialni operator pouzivany hlavne s numpy nebo pri volani funkci
        #pro @ musi byt pocet sloupcu v prvnim array stejny jako pocet radku v druhem array
        #(2, 2) @ (1, 2) = (2, 1) nebo (2, 3) @ (3, 2) = (2, 2)
        h_pre = b_i_h + w_i_h @ image
        #h_pre (pre jakoze neni finalni) ale muze byt od minusove hodnoty az po hodnotu vetsi nez jedna
        #musime tedy hodnotu normalizovat mezi 0 az 1
        #na to se pouziva tzv. aktivacni funkce, tech je hodne ale my pouzijeme sigmoid
        h = 1 / (1 + np.exp(-h_pre)) #funkci je pouze o matematice a vyber je velky

        #forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        nr_correct += int(np.argmax(o) == np.argmax(label))

        #aby se nase NN mohla vytrenovat a uzpusobit weights a biases pro maximalni presnost
        #na to se vypocita rozdil mezi outputem a jeho labelem
        delta_o = o - label
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(image)
        b_i_h += -learn_rate * delta_h

    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0


#nas vytrenovany model ktery muzeme pouzivat jako cli aplikaci
while True:
    try:
        #input indexu obrazku z naseho testovaciho datasetu
        index = int(input("Enter a number (0 - 10 000): "))
        img = images_test[index]
        #ukazani obrazku
        plt.imshow(img.reshape(28, 28), cmap="Greys")

        #ted nas obrazek projede nasi neuronovou siti tentokrat uz ale se spravnymi weights a biases
        img.shape += (1,)
        # Forward propagation input -> hidden
        h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
        h = 1 / (1 + np.exp(-h_pre))
        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        plt.title("The image")
        print(f"The image shows: {o.argmax()}")
        plt.show()
    except:
        print("Error. Enter a number")
        continue


#zkuste zadat cislo 543 na ruzny pocet epoch, jak program rozezna tento obrazek pri jednom nebo deseti epochy?